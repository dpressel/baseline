import math
import logging
import torch
import torch.autograd
from eight_mile.optz import create_lr_scheduler, register_lr_scheduler
from eight_mile.optz import (
    ConstantScheduler,
    WarmupLinearScheduler,
    CyclicLRScheduler,
    PiecewiseDecayScheduler,
    ZarembaDecayScheduler,
    CosineDecayScheduler,
    InverseTimeDecayScheduler,
    ExponentialDecayScheduler,
    CompositeLRScheduler,
    LinearDecayScheduler,
)

logger = logging.getLogger("mead.layers")


@register_lr_scheduler(name="default")
class ConstantSchedulerPyTorch(ConstantScheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@register_lr_scheduler(name="warmup_linear")
class WarmupLinearSchedulerPyTorch(WarmupLinearScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

@register_lr_scheduler(name="linear")
class LinearDecaySchedulerPyTorch(LinearDecayScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

@register_lr_scheduler(name="clr")
class CyclicLRSchedulerPyTorch(CyclicLRScheduler):
    def __init__(self, *args, **kwargs):
        super().__init(*args, **kwargs)


@register_lr_scheduler(name="piecewise")
class PiecewiseDecaySchedulerPyTorch(PiecewiseDecayScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_lr_scheduler(name="zaremba")
class ZarembaDecaySchedulerPyTorch(ZarembaDecayScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_lr_scheduler(name="cosine")
class CosineDecaySchedulerPyTorch(CosineDecayScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_lr_scheduler(name="invtime")
class InverseTimeDecaySchedulerPytorch(InverseTimeDecayScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_lr_scheduler(name="exponential")
class ExponentialDecaySchedulerPyTorch(ExponentialDecayScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_lr_scheduler(name="composite")
class CompositeLRSchedulerPyTorch(CompositeLRScheduler):
    pass


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                if group["weight_decay"] != 0.0:
                    p.data.add_(p.data, alpha=-group["weight_decay"] * group["lr"])

        return loss


class OptimizerManager:
    def __init__(self, model_or_params, global_step=0, **kwargs):
        if isinstance(model_or_params, torch.nn.Module):
            parameters = model_or_params.parameters()
        else:
            parameters = model_or_params
        self.global_step = global_step
        if "lr_function" in kwargs:
            self.lr_function = kwargs["lr_function"]
        else:
            if "lr_scheduler_type" not in kwargs:
                kwargs["lr_scheduler_type"] = "default"
            self.lr_function = create_lr_scheduler(**kwargs)
        self._init_optimizer(parameters, **kwargs)
        self.current_lr = 0

    @property
    def global_step(self):
        return self._global_step

    @global_step.setter
    def global_step(self, value):
        self._global_step = value

    def _init_optimizer(self, parameters, **kwargs):
        wd = float(kwargs.get("weight_decay", 0))
        optim = kwargs.get("optim", "sgd")
        self.current_lr = kwargs.get("eta", kwargs.get("lr", 0.01))
        if optim == "adadelta":
            logger.info("adadelta(eta=%f, wd=%f)", self.current_lr, wd)
            self.optimizer = torch.optim.Adadelta(parameters, lr=self.current_lr, weight_decay=wd)
        elif optim.startswith("adam"):
            beta1 = kwargs.get("beta1", 0.9)
            beta2 = kwargs.get("beta2", 0.999)
            eps = kwargs.get("epsilon", 1e-8)
            if optim == "adam":
                logger.info(
                    "adam(eta=%f, beta1=%f, beta2=%f, epsilon=%f, wd=%f)", self.current_lr, beta1, beta2, eps, wd
                )
                self.optimizer = torch.optim.Adam(
                    parameters, lr=self.current_lr, betas=(beta1, beta2), eps=eps, weight_decay=wd
                )
            elif optim == "adamw":
                logger.info(
                    "adamw(eta=%f, beta1=%f, beta2=%f, epsilon=%f, wd=%f)", self.current_lr, beta1, beta2, eps, wd
                )
                self.optimizer = AdamW(
                    parameters,
                    lr=self.current_lr,
                    betas=(beta1, beta2),
                    eps=eps,
                    weight_decay=wd,
                )
        elif optim == "rmsprop":
            mom = kwargs.get("mom", 0.0)
            logger.info("rmsprop(eta=%f, wd=%f, mom=%f)", self.current_lr, wd, mom)
            self.optimizer = torch.optim.RMSprop(parameters, lr=self.current_lr, weight_decay=wd, momentum=mom)
        elif optim == "asgd":
            logger.info("asgd(eta=%f, wd=%f)", self.current_lr, wd)
            self.optimizer = torch.optim.ASGD(parameters, lr=self.current_lr, weight_decay=wd)
        else:
            mom = kwargs.get("mom", 0.9)
            logger.info("sgd(eta=%f, mom=%f, wd=%f)", self.current_lr, mom, wd)
            self.optimizer = torch.optim.SGD(parameters, lr=self.current_lr, momentum=mom, weight_decay=wd)

    def _identity(self, _):
        return self.current_lr

    def step(self):
        """Runs at every step and updates the learning rate

        :return:
        """
        self.optimizer.step()
        self.current_lr = self.update_lr()
        self.global_step += 1

    def scale_grads(self, scalar):
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                if p.grad is not None:
                    p.grad.data.mul_(scalar)

    def clip_grads(self, grad_clip):
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                if p.grad is not None:
                    p.grad.clamp_(-grad_clip, grad_clip)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update_lr(self):
        lr = self.lr_function(self.global_step)
        for p in self.optimizer.param_groups:
            p["lr"] = lr

        return lr


class EagerOptimizer:
    def __init__(self, loss, optimizer=None, **kwargs):
        self.loss = loss
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer_args = kwargs
            self.optimizer = None

    def update(self, model, x, y):
        if not self.optimizer:
            self.optimizer = OptimizerManager(model, **self.optimizer_args)
        self.optimizer.zero_grad()
        l = self.loss(model, x, y)
        l.backward()
        self.optimizer.step()
        return float(l)

    def update_with_hidden(self, model, h, x, y):
        if not self.optimizer:
            self.optimizer = OptimizerManager(model, **self.optimizer_args)
        self.optimizer.zero_grad()
        l, h = self.loss(model, h, x, y)
        l.backward()
        self.optimizer.step()
        return float(l), h
