"""Training loop utilities

This provides an interface that makes it easy to wire up hooks and train over multiple devices.
To use these utilities, define a `TrainingTarget` with a stepwise definition for training and
validation.


"""
import tensorflow as tf
import os
import numpy as np
import logging
from copy import deepcopy
from typing import Union
from typing import Optional, Dict, List, Tuple
from eight_mile.tf.layers import SET_TRAIN_FLAG, TRAIN_FLAG, create_distribute_strategy
from eight_mile.tf.optz import OptimizerManager
from eight_mile.progress import create_progress_bar
from eight_mile.confusion import ConfusionMatrix
from eight_mile.utils import Average, get_num_gpus_multiworker, revlut, listify
from contextlib import ExitStack
from baseline.utils import get_metric_cmp
from baseline.embeddings import *
logger = logging.getLogger(__file__)


class TrainingTarget(tf.keras.Model):

    def __init__(self, name=None):
        super().__init__(name=name)

    def train_step(self, batch, device):
        pass

    def eval_step(self, batch, device):
        pass

    @property
    def model(self):
        pass


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
                elif metric == 'confusion':
                    self.metrics[metric] = ConfusionMatrix(np.arange(len(local_metrics[metric])))
                else:
                    self.metrics[metric] = Average(metric)

            if isinstance(local_metrics[metric], tf.Tensor):
                tensor = local_metrics[metric].numpy()
            else:
                tensor = local_metrics[metric]
            self.metrics[metric].update(tensor)

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
        self.step2checkpoint = {}

    def run(self, model, metrics, global_step):
        checkpoint = tf.train.Checkpoint(model=model)
        fname = checkpoint.save(os.path.join(self.checkpoint_dir, self.model_base))
        self.step2checkpoint[global_step] = fname

    def get_model_file(self, global_step):
        return self.step2checkpoint[global_step]


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

    def restore(self, module, global_step=-1):
        checkpoint = tf.train.Checkpoint(model=module)
        checkpoint.restore(self.get_model_file(global_step))


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
        print('subs', self.train_module.submodules)
        print('vars', self.train_module.variables)
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
            report_on: int = 100,
            early_stopping_metric: str=None,
            local_rank=0,
            distributed=False,
            basedir: str=None,
            max_steps_per_epoch=None,
            strategy_type='mirror',
            eval_device='/device:GPU:0',
            endpoint=None,
            num_gpus=1,
            progress_bar='default'):

        # Get the basedir to save results and checkpoints
        if basedir is None:
            basedir = f'checkpoints-{os.getpid()}'
        os.makedirs(basedir, exist_ok=True)

        # Setup logger
        logging.basicConfig(level=logging.INFO if local_rank in [-1, 0] else logging.WARN)

        distributed = distributed or num_gpus > 1
        logger.info(f"Using {num_gpus} GPUs in this job.")

        if distributed and strategy_type == 'tpu':
            eval_device = '/device:CPU:0'
            num_gpus = 0

        strategy = create_distribute_strategy(local_rank, endpoint, num_gpus)

        eval_strategy = tf.distribute.OneDeviceStrategy(eval_device)

        checkpoint_manager = CheckpointManager(basedir, early_stopping_key=early_stopping_metric)
        self.valid_metric_observers.append(checkpoint_manager)

        steps_train = len(train_loader)
        steps_valid = len(valid_loader) if valid_loader else 0
        steps_eval = len(eval_loader) if eval_loader else 0

        if distributed:
            train_loader = strategy.experimental_distribute_dataset(train_loader)

        @tf.function
        def distributed_train_step(inputs):

            def _replicated_train_step(replicated_inputs):

                with tf.GradientTape() as tape:
                    metrics = self.train_module.train_step(replicated_inputs)
                    loss = metrics[self.loss_key]
                    self.optimizer.step(tape, loss)
                return metrics
            metrics = strategy.run(_replicated_train_step, args=(inputs,))
            for metric in metrics:
                metrics[metric] = strategy.reduce(tf.distribute.ReduceOp.SUM, metrics[metric], axis=None)
            return metrics

        @tf.function
        def distributed_eval_step(inputs):
            metrics = strategy.run(self.train_module.eval_step, args=(inputs,))
            for metric in metrics:
                metrics[metric] = strategy.reduce(tf.distribute.ReduceOp.SUM, metrics[metric], axis=None)
            return metrics

        for epoch in range(num_epochs):
            with strategy.scope():
                SET_TRAIN_FLAG(True)
                steps = steps_train
                if max_steps_per_epoch and max_steps_per_epoch < steps_train:
                    steps = max_steps_per_epoch
                pg = create_progress_bar(steps, name=progress_bar)
                epoch_train_metrics = GlobalMetrics()
                last_report_step = -1

                for iters, batch in enumerate(pg(train_loader)):
                    metrics = distributed_train_step(batch)
                    epoch_train_metrics.update(metrics)

                    if self.optimizer.global_step % report_on == 0:
                        last_report_step = self.optimizer.global_step
                        self._fire_train_observers(epoch_train_metrics.reduce())

                if steps_valid < 1 or local_rank > 0:
                    continue

                if self.optimizer.global_step != last_report_step:
                    self._fire_train_observers(epoch_train_metrics.reduce())

                #with eval_strategy.scope():

                SET_TRAIN_FLAG(False)
                pg = create_progress_bar(steps_valid)
                epoch_valid_metrics = GlobalMetrics()
                for batch in pg(valid_loader):
                    metrics = distributed_eval_step(batch)
                    epoch_valid_metrics.update(metrics)
                self._fire_valid_observers(epoch_valid_metrics.reduce())

        if steps_eval < 1 or local_rank > 0:
            return

        with eval_strategy.scope():
            pg = create_progress_bar(steps_eval)
            epoch_eval_metrics = GlobalMetrics()
            SET_TRAIN_FLAG(False)
            checkpoint_manager.restore(self.train_module)
            for batch in pg(eval_loader):
                metrics = distributed_eval_step(batch)
                epoch_eval_metrics.update(metrics)

            self._fire_test_observers(epoch_eval_metrics.reduce())
