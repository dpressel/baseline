"""Training loop utilities

This provides an interface that makes it easy to wire up hooks and train over multiple devices.
To use these utilities, define a `TrainingTarget` with a stepwise definition for training and
validation.


"""

import torch
import torch.nn as nn
import os
import numpy as np
import logging
from copy import deepcopy
from typing import Union
from typing import Optional, Dict, List, Tuple
from eight_mile.pytorch.optz import OptimizerManager
from eight_mile.pytorch.layers import save_checkpoint, rm_old_checkpoints, checkpoint_for, init_distributed
from eight_mile.progress import create_progress_bar
from eight_mile.confusion import ConfusionMatrix
from eight_mile.utils import Average, get_num_gpus_multiworker, revlut, listify
from eight_mile.pytorch.layers import FineTuneModel
from contextlib import ExitStack
from baseline.utils import get_metric_cmp
from baseline.embeddings import *
from baseline.pytorch.embeddings import TransformerLMPooledEmbeddingsModel
from baseline.vectorizers import BPEVectorizer1D
from baseline.reader import TSVSeqLabelReader
logger = logging.getLogger(__file__)


class TrainingTarget(nn.Module):

    def __init__(self):
        super().__init__()
        self.device = 'cpu'

    def train_step(self, batch, device):
        pass

    def eval_step(self, batch, device):
        pass

    def load_checkpoint(self, checkpoint_name):
        pass

    def save_checkpoint(self, checkpoint_name):
        pass

    @property
    def model(self):
        pass

    def set_device(self, device):
        self.to(device=device)
        self.device = device


class GlobalMetrics:

    def __init__(self):
        self.metrics = {}

    def reduce(self):
        metrics = {}
        for metric in self.metrics.keys():
            if isinstance(self.metrics[metric], ConfusionMatrix):
                all_metrics = self.metrics[metric].get_all_metrics()
                for cm_metric in all_metrics:
                    metrics[cm_metric] = all_metrics[cm_metric]

            else:
                metrics[metric] = self.metrics[metric].avg
        return metrics


    def update(self, local_metrics):
        for metric in local_metrics:
            if metric not in self.metrics:
                if isinstance(local_metrics[metric], ConfusionMatrix):
                    self.metrics[metric] = ConfusionMatrix(local_metrics[metric].labels)
                else:
                    self.metrics[metric] = Average(metric)

            self.metrics[metric].update(local_metrics[metric])

    def __getitem__(self, item):
        return self.metrics[item]


    def __setitem__(self, key, value):
        self.metrics[key] = value

    def __len__(self):
        return len(self.metrics)

    def keys(self):
        return self.metrics.keys()

    def items(self):
        return self.metrics.items()

    def values(self):
        return self.metrics.values()


class MetricObserver:

    def run(self, model, metrics, global_step):
        pass


class LogAllMetrics(MetricObserver):

    def __init__(self, name):
        self.name = name

    def run(self, model, metrics, global_step):
        name = f"[{self.name}]"
        print(f"\n{name:>14}: {global_step}")
        for metric in metrics.keys():
            print(f"{metric:>14}: {metrics[metric]}")


class SaveCheckpoint(MetricObserver):
    def __init__(self, checkpoint_dir, model_base='checkpoint'):
        self.checkpoint_dir = checkpoint_dir
        self.model_base = os.path.join(self.checkpoint_dir, model_base)

    def run(self, model, metrics, global_step):
        #rm_old_checkpoints(self.checkpoint_dir)
        save_checkpoint(model, self.model_base, global_step, tick_type='step')

    def get_model_file(self, global_step):
        return checkpoint_for(self.model_base, global_step, tick_type='step') + '.pth'


class CheckpointManager(MetricObserver):
    # This doesnt actually have to save if we can guarantee there is a save metric in there somewhere
    def __init__(self, checkpoint_dir, model_base='checkpoint', early_stopping_key=None):
        self.early_stopping_key = early_stopping_key
        self.saver = SaveCheckpoint(checkpoint_dir, model_base)
        if self.early_stopping_key:
            self.early_stopping_cmp, self.best_metric = get_metric_cmp(self.early_stopping_key)
        self.step = -1

    def run(self, model, metrics, global_step):
        self.saver.run(model, metrics, global_step)
        if self.early_stopping_key:
            current = metrics[self.early_stopping_key]
            if self.early_stopping_cmp(current, self.best_metric):
                self.step = global_step
                self.best_metric = current
                logger.info('New best %.3f', self.best_metric)
        else:
            self.step = global_step

    def get_model_file(self, global_step=-1):
        if global_step < 1:
            global_step = self.step
        return self.saver.get_model_file(global_step)

    def restore(self, module, global_step=-1, str_map={}, map_location=None):
        ckpt_dict = torch.load(self.get_model_file(global_step), map_location=map_location)
        renamed = {}
        for k, v in ckpt_dict.items():
            for from_str, to_str in str_map.items():
                k = k.replace(from_str, to_str)
            renamed[k] = v
        unmatch = module.load_state_dict(renamed, strict=False)
        if unmatch.missing_keys or len(unmatch.unexpected_keys) > 2:
            print("Warning: Embedding doesn't match with the checkpoint being loaded.")
            print(f"missing keys: {unmatch.missing_keys}\n unexpected keys: {unmatch.unexpected_keys}")


class Trainer:

    def __init__(
            self,
            train_module,
            global_step: int = 0,
            optim: str='adam',
            lr: float=0.001,
            weight_decay: float=0.0,
            loss_key: str = 'loss',
            clip: float=50.0,
            train_metric_observers: Union[List[MetricObserver], MetricObserver] = [],
            valid_metric_observers: Union[List[MetricObserver], MetricObserver] = [],
            test_metric_observers: Union[List[MetricObserver], MetricObserver] = [],
            **kwargs):
        self.train_module = train_module
        self.clip = clip
        self.optimizer = OptimizerManager(train_module, global_step, optim=optim, lr=lr, weight_decay=weight_decay)
        self.loss_key = loss_key
        self.train_metric_observers = listify(train_metric_observers)
        self.valid_metric_observers = listify(valid_metric_observers)
        self.test_metric_observers = listify(test_metric_observers)

    def _fire_train_observers(self, metrics):
        for observer in self.train_metric_observers:
            observer.run(self.train_module, metrics, self.optimizer.global_step)

    def _fire_valid_observers(self, metrics):
        for observer in self.valid_metric_observers:
            observer.run(self.train_module, metrics, self.optimizer.global_step)

    def _fire_test_observers(self, metrics):
        for observer in self.test_metric_observers:
            observer.run(self.train_module, metrics, self.optimizer.global_step)

    def run(self, train_loader, valid_loader=None, eval_loader=None, num_epochs: int = 1,
            report_on: int = 100, grad_accum: int = 1,
            early_stopping_metric: str=None,
            local_rank=0,
            distributed=False,
            basedir: str=None,
            device='cuda',
            max_steps_per_epoch=None,
            progress_bar='default'):

        # Get the basedir to save results and checkpoints
        if basedir is None:
            basedir = f'checkpoints-{os.getpid()}'
        os.makedirs(basedir, exist_ok=True)

        # Setup logger
        logging.basicConfig(level=logging.INFO if local_rank in [-1, 0] else logging.WARN)
        num_gpus = get_num_gpus_multiworker()

        distributed = distributed or num_gpus > 1
        logger.info(f"Using {num_gpus} GPUs in this job.")

        if distributed:
            device, updated_local_rank = init_distributed(local_rank)
            local_rank = updated_local_rank

        self.train_module.set_device(device)
        checkpoint_manager = CheckpointManager(basedir, early_stopping_key=early_stopping_metric)
        self.valid_metric_observers.append(checkpoint_manager)

        steps_train = len(train_loader)
        steps_valid = len(valid_loader) if valid_loader else 0
        steps_eval = len(eval_loader) if eval_loader else 0

        for epoch in range(num_epochs):
            self.train_module.train()
            steps = steps_train
            if max_steps_per_epoch and max_steps_per_epoch < steps_train:
                steps = max_steps_per_epoch
            pg = create_progress_bar(steps, name=progress_bar)
            epoch_train_metrics = GlobalMetrics()
            last_report_step = -1

            for iters, batch in enumerate(pg(train_loader)):
                is_dist_step = iters % grad_accum == 0
                with self.train_module.no_sync() if (distributed and not is_dist_step) else ExitStack():
                    metrics = self.train_module.train_step(batch)
                    loss = metrics[self.loss_key]
                    epoch_train_metrics.update(metrics)
                    loss.backward()

                    if is_dist_step:
                        if self.clip and self.clip > 0.0:
                            self.optimizer.clip_grads(self.clip)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        if max_steps_per_epoch is not None and (iters / grad_accum) >= (max_steps_per_epoch-1):
                            break

                        if self.optimizer.global_step % report_on == 0:
                            last_report_step = self.optimizer.global_step
                            self._fire_train_observers(epoch_train_metrics.reduce())

            if steps_valid < 1 or local_rank > 0:
                continue

            if self.optimizer.global_step != last_report_step:
                self._fire_train_observers(epoch_train_metrics.reduce())

            self.train_module.eval()
            pg = create_progress_bar(steps_valid)
            epoch_valid_metrics = GlobalMetrics()
            for batch in pg(valid_loader):
                metrics = self.train_module.eval_step(batch)
                epoch_valid_metrics.update(metrics)
            self._fire_valid_observers(epoch_valid_metrics.reduce())

        if steps_eval < 1 or local_rank > 0:
            return

        pg = create_progress_bar(steps_eval)
        epoch_eval_metrics = GlobalMetrics()

        checkpoint_manager.restore(self.train_module, map_location=device)
        self.train_module.eval()
        for batch in pg(eval_loader):
            metrics = self.train_module.eval_step(batch)
            epoch_eval_metrics.update(metrics)

        self._fire_test_observers(epoch_eval_metrics.reduce())
