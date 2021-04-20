import logging
import tensorflow as tf
from eight_mile.utils import get_version, exporter, register
from eight_mile.optz import create_lr_scheduler, register_lr_scheduler, MEAD_LAYERS_LR_SCHEDULERS
from eight_mile.optz import (
    LearningRateScheduler,
    ConstantScheduler,
    WarmupLearningRateScheduler,
    WarmupLinearScheduler,
    CyclicLRScheduler,
    PiecewiseDecayScheduler,
    ZarembaDecayScheduler,
    CosineDecayScheduler,
)

logger = logging.getLogger("mead.layers")


class ConstantSchedulerTensorFlow1:
    def __init__(self, **kwargs):
        pass

    def __call__(self, lr, global_step):
        return tf.identity(lr, name="lr")

    def __str__(self):
        return type(self).__name__ + "()"


class WarmupLinearSchedulerTensorFlow1(WarmupLearningRateScheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, lr, global_step):
        return tf.identity(
            tf.minimum(1.0, tf.cast(global_step / self.warmup_steps, dtype=tf.float32)) * lr, name="lr"
        )

    def __str__(self):
        return type(self).__name__ + "()"


class CyclicLRSchedulerTensorFlow1:
    def __init__(self, max_lr=1e-2, decay_steps=1000, **kwargs):
        super().__init__()
        self.max_lr = max_lr
        self.decay_steps = decay_steps

    def __call__(self, lr, global_step):
        gs_f = tf.cast(global_step, tf.float32)
        cycle = tf.floor(1.0 + gs_f / (2.0 * self.decay_steps))
        x = tf.abs(gs_f / self.decay_steps - 2.0 * cycle + 1.0)
        clr = lr + (self.max_lr - lr) * tf.maximum(0.0, 1.0 - x)
        return tf.identity(clr, name="lr")

    def __str__(self):
        return type(self).__name__ + "()"


class SGDRSchedulerTensorFlow1:
    def __init__(self, first_decay_steps=1000, **kwargs):
        super().__init__()
        self.first_decay_steps = first_decay_steps

    def __call__(self, lr, global_step):
        return tf.identity(
            tf.train.cosine_decay_restarts(lr, global_step, first_decay_steps=self.first_decay_steps), name="lr"
        )

    def __str__(self):
        return type(self).__name__ + "()"


class CompositeLRSchedulerTensorFlow1:
    def __init__(self, warm=None, rest=None, **kwargs):
        self.warm = warm
        self.rest = rest

    def __call__(self, lr, global_step):
        warm_tensor = self.warm(lr, global_step)

        def call_warm():
            return warm_tensor

        rest_step = tf.subtract(global_step, tf.compat.v1.constant(self.warm.warmup_steps, dtype=global_step.dtype))
        rest_tensor = self.rest(lr, rest_step)

        def call_rest():
            return rest_tensor

        return tf.identity(tf.cond(global_step < self.warm.warmup_steps, call_warm, call_rest), name="lr")

    def __str__(self):
        return "LRScheduler({}, {})".format(self.warm, self.rest)


class PiecewiseDecaySchedulerTensorFlow1:
    def __init__(self, boundaries=None, values=None, **kwargs):
        super().__init__()
        self.boundaries = boundaries
        self.values = values

    def __call__(self, lr, global_step):
        return tf.identity(tf.compat.v1.train.piecewise_constant(global_step, self.boundaries, self.values), name="lr")

    def __str__(self):
        return type(self).__name__ + "()"


class InverseTimeDecaySchedulerTensorFlow1:
    def __init__(self, decay_steps=16000, decay_rate=0.05, staircase=False, **kwargs):
        super().__init__()
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def __call__(self, lr, global_step):
        return tf.identity(
            tf.train.inverse_time_decay(
                lr, global_step, self.decay_steps, self.decay_rate, staircase=self.staircase
            ),
            name="lr",
        )

    def __str__(self):
        return type(self).__name__ + "()"


class ExponentialDecaySchedulerTensorFlow1:
    def __init__(self, decay_steps=16000, decay_rate=0.5, staircase=False, **kwargs):
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def __call__(self, lr, global_step):
        return tf.identity(
            tf.train.exponential_decay(
                lr, global_step, self.decay_steps, self.decay_rate, staircase=self.staircase
            ),
            name="lr",
        )

    def __str__(self):
        return type(self).__name__ + "()"


class ZarembaDecaySchedulerTensorFlow1(PiecewiseDecaySchedulerTensorFlow1):
    """Utility only, just to simplify the JSON"""

    def __init__(self, boundaries=None, decay_rate=None, **kwargs):
        lr = float(kwargs.get("lr", kwargs.get("eta", 1.0)))
        values = [lr / (float(decay_rate) ** i) for i in range(len(boundaries) + 1)]
        super().__init__(boundaries=boundaries, values=values)

    def __str__(self):
        return type(self).__name__ + "()"


class ConstantSchedulerTensorFlow2(LearningRateScheduler, tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, global_step):
        tf.summary.scalar(name='lr', data=self.lr, step=tf.cast(global_step, tf.int64))
        return self.lr

    def __str__(self):
        return type(self).__name__ + "()"


class WarmupLinearSchedulerTensorFlow2(
    WarmupLearningRateScheduler, tf.keras.optimizers.schedules.LearningRateSchedule
):
    def __init__(self, warmup_steps=16000, **kwargs):
        lr = float(kwargs.get("lr", kwargs.get("eta", 1.0)))
        kwargs["lr"] = lr
        super().__init__(warmup_steps=warmup_steps, **kwargs)

    def __call__(self, global_step):
        new_lr = tf.minimum(1.0, global_step / float(self.warmup_steps)) * self.lr
        tf.summary.scalar(name='warmup_lr', data=new_lr, step=tf.cast(global_step, tf.int64))
        return new_lr

    def __str__(self):
        return type(self).__name__ + "()"


class CyclicLRSchedulerTensorFlow2(LearningRateScheduler, tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr=1e-2, decay_steps=1000, **kwargs):
        lr = float(kwargs.get("lr", kwargs.get("eta", 1.0)))
        kwargs["lr"] = lr
        super().__init__(**kwargs)
        self.max_lr = max_lr
        self.decay_steps = decay_steps

    def __call__(self, global_step):
        gs_f = tf.cast(global_step, tf.float32)
        cycle = tf.floor(1.0 + gs_f / (2.0 * self.decay_steps))
        x = tf.abs(gs_f / self.decay_steps - 2.0 * cycle + 1.0)
        clr = self.lr + (self.max_lr - self.lr) * tf.maximum(0.0, 1.0 - x)
        new_lr = tf.identity(clr, name="lr")
        tf.summary.scalar(name='clr_lr', data=new_lr, step=tf.cast(global_step, tf.int64))
        return new_lr

    def __str__(self):
        return type(self).__name__ + "()"


class SGDRSchedulerTensorFlow2(LearningRateScheduler, tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, first_decay_steps=1000, **kwargs):
        super().__init__(**kwargs)
        self.first_decay_steps = first_decay_steps

    def __call__(self, global_step):
        new_lr = tf.identity(
            tf.compat.v1.train.cosine_decay_restarts(
                self.lr, global_step, first_decay_steps=self.first_decay_steps
            ),
            name="lr",
        )
        tf.summary.scalar(name='sgdr_lr', data=new_lr, step=tf.cast(global_step, tf.int64))
        return new_lr

    def __str__(self):
        return type(self).__name__ + "()"


class ZarembaDecaySchedulerTensorFlow2(tf.keras.optimizers.schedules.PiecewiseConstantDecay):
    """Utility only, just to simplify the JSON"""

    def __init__(self, boundaries=None, decay_rate=None, **kwargs):
        lr = float(kwargs.get("lr", kwargs.get("eta", 1.0)))
        values = [lr / (float(decay_rate) ** i) for i in range(len(boundaries) + 1)]
        super().__init__(boundaries, values, kwargs.get("name"))


class CompositeLRSchedulerTensorFlow2(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warm=None, rest=None, plateau_steps=0, **kwargs):
        self.warm = warm
        self.rest = rest
        self.plateau_steps = plateau_steps

    def __call__(self, global_step):
        total_steps_lr1 = self.warm.warmup_steps + self.plateau_steps
        warm_tensor = self.warm(global_step)

        def call_warm():
            return warm_tensor

        rest_step = global_step - total_steps_lr1
        rest_tensor = self.rest(rest_step)

        def call_rest():
            return rest_tensor

        new_lr = tf.cond(global_step < total_steps_lr1, call_warm, call_rest)
        tf.summary.scalar(name='composite_lr', data=new_lr, step=tf.cast(global_step, tf.int64))
        return new_lr

    def __str__(self):
        return "LRScheduler({}, {})".format(self.warm, self.rest)


class PiecewiseDecaySchedulerTensorFlow2(tf.keras.optimizers.schedules.PiecewiseConstantDecay):
    def __init__(self, boundaries, values, **kwargs):
        super().__init__(boundaries, values)

    def __call__(self, global_step):
        new_lr = super().__call__(global_step)
        tf.summary.scalar(name='piecewise_decay_lr', data=new_lr, step=tf.cast(global_step, tf.int64))
        return new_lr


class InverseTimeDecaySchedulerTensorFlow2(tf.keras.optimizers.schedules.InverseTimeDecay):
    def __init__(self, decay_steps=16000, decay_rate=0.05, staircase=False, **kwargs):
        lr = kwargs.get("lr", kwargs.get("eta", 0.01))
        super().__init__(lr, decay_steps, decay_rate, staircase, kwargs.get("name"))

    def __call__(self, global_step):
        new_lr = super().__call__(global_step)
        tf.summary.scalar(name='inv_time_decay_lr', data=new_lr, step=tf.cast(global_step, tf.int64))
        return new_lr


class ExponentialDecaySchedulerTensorFlow2(tf.keras.optimizers.schedules.ExponentialDecay):
    def __init__(self, decay_steps=16000, decay_rate=0.5, staircase=False, **kwargs):
        lr = kwargs.get("lr", kwargs.get("eta", 0.01))
        super().__init__(lr, decay_steps, decay_rate, staircase, kwargs.get("name"))

    def __call__(self, global_step):
        new_lr = super().__call__(global_step)
        tf.summary.scalar(name='exponential_decay_lr', data=new_lr, step=tf.cast(global_step, tf.int64))
        return new_lr


class CosineDecaySchedulerTensorFlow(CosineDecayScheduler, tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, decay_steps=16000, alpha=0.0, **kwargs):
        kwargs['lr'] = kwargs.get("lr", kwargs.get("eta", 0.01))
        super().__init__(decay_steps, alpha, **kwargs)

    def __call__(self, global_step):
        global_step = tf.math.minimum(global_step, self.decay_steps)
        cosine_decay = 0.5 * (1.0 + tf.cos(3.14159265 * global_step / self.decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        new_lr = self.lr * decayed
        tf.summary.scalar(name='cosine_decay_lr', data=new_lr, step=tf.cast(global_step, tf.int64))
        return new_lr


class LinearDecaySchedulerTensorFlow(LearningRateScheduler, tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, decay_steps=1000, alpha=0.0, **kwargs):
        super().__init__(**kwargs)
        self.decay_steps = decay_steps
        self.alpha = alpha

    def __call__(self, global_step):
        global_step = min(global_step, self.decay_steps)
        # Linear interpolation
        scaled_lr = self.lr * (1.0 - self.alpha) * (1.0 - global_step / self.decay_steps) + (self.alpha * self.lr)
        return scaled_lr


if not tf.executing_eagerly():

    ConstantSchedulerTensorFlow = ConstantSchedulerTensorFlow1
    WarmupLinearSchedulerTensorFlow = WarmupLinearSchedulerTensorFlow1
    CyclicLRSchedulerTensorFlow = CyclicLRSchedulerTensorFlow1
    SGDRSchedulerTensorFlow = SGDRSchedulerTensorFlow1
    CompositeLRSchedulerTensorFlow = CompositeLRSchedulerTensorFlow1
    PiecewiseDecaySchedulerTensorFlow = PiecewiseDecaySchedulerTensorFlow1
    ExponentialDecaySchedulerTensorFlow = ExponentialDecaySchedulerTensorFlow1
    InverseTimeDecaySchedulerTensorFlow = InverseTimeDecaySchedulerTensorFlow1
    ZarembaDecaySchedulerTensorFlow = ZarembaDecaySchedulerTensorFlow1
else:
    ConstantSchedulerTensorFlow = ConstantSchedulerTensorFlow2
    WarmupLinearSchedulerTensorFlow = WarmupLinearSchedulerTensorFlow2
    CyclicLRSchedulerTensorFlow = CyclicLRSchedulerTensorFlow2
    SGDRSchedulerTensorFlow = SGDRSchedulerTensorFlow2
    CompositeLRSchedulerTensorFlow = CompositeLRSchedulerTensorFlow2
    PiecewiseDecaySchedulerTensorFlow = PiecewiseDecaySchedulerTensorFlow2
    ExponentialDecaySchedulerTensorFlow = ExponentialDecaySchedulerTensorFlow2
    InverseTimeDecaySchedulerTensorFlow = InverseTimeDecaySchedulerTensorFlow2
    ZarembaDecaySchedulerTensorFlow = ZarembaDecaySchedulerTensorFlow2
    register(LinearDecaySchedulerTensorFlow, MEAD_LAYERS_LR_SCHEDULERS, "linear", "lr_scheduler")

register(ConstantSchedulerTensorFlow, MEAD_LAYERS_LR_SCHEDULERS, "default", "lr_scheduler")
register(WarmupLinearSchedulerTensorFlow, MEAD_LAYERS_LR_SCHEDULERS, "warmup_linear", "lr_scheduler")
register(CyclicLRSchedulerTensorFlow, MEAD_LAYERS_LR_SCHEDULERS, "clr", "lr_scheduler")
register(SGDRSchedulerTensorFlow, MEAD_LAYERS_LR_SCHEDULERS, "sgdr", "lr_scheduler")
register(CompositeLRSchedulerTensorFlow, MEAD_LAYERS_LR_SCHEDULERS, "composite", "lr_scheduler")
register(PiecewiseDecaySchedulerTensorFlow, MEAD_LAYERS_LR_SCHEDULERS, "piecewise", "lr_scheduler")
register(ZarembaDecaySchedulerTensorFlow, MEAD_LAYERS_LR_SCHEDULERS, "zaremba", "lr_scheduler")
register(CosineDecaySchedulerTensorFlow, MEAD_LAYERS_LR_SCHEDULERS, "cosine", "lr_scheduler")
register(InverseTimeDecaySchedulerTensorFlow, MEAD_LAYERS_LR_SCHEDULERS, "invtime", "lr_scheduler")
register(ExponentialDecaySchedulerTensorFlow, MEAD_LAYERS_LR_SCHEDULERS, "exponential", "lr_scheduler")


class AdamWOptimizer(tf.compat.v1.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay.

    Modified from: https://github.com/google-research/bert/blob/master/optimization.py
    This does the weight decay slightly differently from PyTorch version, putting it before the update
    """

    def __init__(self, learning_rate, weight_decay=0.0, beta_1=0.9, beta_2=0.999, epsilon=1e-6, name="AdamWOptimizer"):
        """Constructs a AdamWOptimizer."""
        super().__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def _get_variable_name(self, param_name):
        import re

        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.compat.v1.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.compat.v1.zeros_initializer(),
            )
            v = tf.compat.v1.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.compat.v1.zeros_initializer(),
            )

            # Standard Adam update.
            next_m = tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad)
            next_v = tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2, tf.square(grad))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)
            update += self.weight_decay * param
            update_with_lr = self.learning_rate * update
            next_param = param - update_with_lr

            assignments.extend([param.assign(next_param), m.assign(next_m), v.assign(next_v)])
        return tf.group(*assignments, name=name)


OPTIMIZER_SUMMARIES = ["learning_rate", "loss", "gradients", "gradient_norm", "global_gradient_norm"]


def _optimize_loss(
    loss,
    global_step,
    learning_rate,
    optimizer,
    gradient_multipliers=None,
    clip_gradients=None,
    learning_rate_decay_fn=None,
    update_ops=None,
    variables=None,
    name=None,
    summaries=None,
    colocate_gradients_with_ops=False,
    increment_global_step=True,
):
    """Given loss and parameters for optimizer, returns a training op.
    Various ways of passing optimizers include:
    - by string specifying the name of the optimizer. See OPTIMIZER_CLS_NAMES
        for full list. E.g. `optimize_loss(..., optimizer='Adam')`.
    - by function taking learning rate `Tensor` as argument and returning an
        `Optimizer` instance. E.g. `optimize_loss(...,
        optimizer=lambda lr: tf.train.MomentumOptimizer(lr, momentum=0.5))`.
      Alternatively, if `learning_rate` is `None`, the function takes no
      arguments. E.g. `optimize_loss(..., learning_rate=None,
        optimizer=lambda: tf.train.MomentumOptimizer(0.5, momentum=0.5))`.
    - by a subclass of `Optimizer` having a single-argument constructor
        (the argument is the learning rate), such as AdamOptimizer or
        AdagradOptimizer. E.g. `optimize_loss(...,
        optimizer=tf.train.AdagradOptimizer)`.
    - by an instance of a subclass of `Optimizer`.
        E.g., `optimize_loss(..., optimizer=tf.train.AdagradOptimizer(0.5))`.
    Args:
      loss: Scalar `Tensor`.
      global_step: Scalar int `Tensor`, step counter to update on each step
                   unless `increment_global_step` is `False`. If not supplied,
                   it will be fetched from the default graph (see
                   `tf.train.get_global_step` for details). If it has
                   not been created, no step will be incremented with each weight
                   update. `learning_rate_decay_fn` requires `global_step`.
      learning_rate: float or `Tensor`, magnitude of update per each training
                     step. Can be `None`.
      optimizer: string, class or optimizer instance, used as trainer.
                 string should be name of optimizer, like 'SGD',
                   'Adam', 'Adagrad'. Full list in OPTIMIZER_CLS_NAMES constant.
                 class should be sub-class of `tf.Optimizer` that implements
                   `compute_gradients` and `apply_gradients` functions.
                 optimizer instance should be instantiation of `tf.Optimizer`
                   sub-class and have `compute_gradients` and `apply_gradients`
                   functions.
      gradient_noise_scale: float or None, adds 0-mean normal noise scaled by this
                            value.
      gradient_multipliers: dict of variables or variable names to floats.
                            If present, gradients for specified
                            variables will be multiplied by given constant.
      clip_gradients: float, callable or `None`. If float, is provided, a global
        clipping is applied to prevent the norm of the gradient to exceed this
        value. Alternatively, a callable can be provided e.g.: adaptive_clipping.
        This callable takes a `list` of `(gradients, variables)` `tuple`s and
        returns the same thing with the gradients modified.
      learning_rate_decay_fn: function, takes `learning_rate` and `global_step`
                              `Tensor`s, returns `Tensor`.
                              Can be used to implement any learning rate decay
                              functions.
                              For example: `tf.train.exponential_decay`.
                              Ignored if `learning_rate` is not supplied.
      update_ops: list of update `Operation`s to execute at each step. If `None`,
                  uses elements of UPDATE_OPS collection. The order of execution
                  between `update_ops` and `loss` is non-deterministic.
      variables: list of variables to optimize or
                 `None` to use all trainable variables.
      name: The name for this operation is used to scope operations and summaries.
      summaries: List of internal quantities to visualize on tensorboard. If not
                 set, the loss, the learning rate, and the global norm of the
                 gradients will be reported. The complete list of possible values
                 is in OPTIMIZER_SUMMARIES.
      colocate_gradients_with_ops: If True, try colocating gradients with the
                                   corresponding op.
      increment_global_step: Whether to increment `global_step`. If your model
        calls `optimize_loss` multiple times per training step (e.g. to optimize
        different parts of the model), use this arg to avoid incrementing
        `global_step` more times than necessary.
    Returns:
      Training op.
    Raises:
      ValueError: if:
          * `loss` is an invalid type or shape.
          * `global_step` is an invalid type or shape.
          * `learning_rate` is an invalid type or value.
          * `optimizer` has the wrong type.
          * `clip_gradients` is neither float nor callable.
          * `learning_rate` and `learning_rate_decay_fn` are supplied, but no
            `global_step` is available.
          * `gradients` is empty.
    """
    loss = tf.convert_to_tensor(loss)
    if global_step is None:
        global_step = tf.compat.v1.train.get_global_step()

    with tf.compat.v1.variable_scope(name, "OptimizeLoss", [loss, global_step]):
        # Update ops take UPDATE_OPS collection if not provided.
        if update_ops is None:
            update_ops = set(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS))
        # Make sure update ops are ran before computing loss.
        if update_ops:
            with tf.compat.v1.control_dependencies(list(update_ops)):
                loss = tf.identity(loss)

        # Learning rate variable, with possible decay.
        lr = None
        if learning_rate is not None:
            if isinstance(learning_rate, tf.Tensor) and learning_rate.get_shape().ndims == 0:
                lr = learning_rate
            elif isinstance(learning_rate, float):
                if learning_rate < 0.0:
                    raise ValueError("Invalid learning_rate %s.", learning_rate)
                lr = tf.compat.v1.get_variable(
                    "learning_rate", [], trainable=False, initializer=tf.compat.v1.constant_initializer(learning_rate)
                )
            else:
                raise ValueError(
                    "Learning rate should be 0d Tensor or float. "
                    "Got %s of type %s" % (str(learning_rate), str(type(learning_rate)))
                )
        if summaries is None:
            summaries = ["loss", "learning_rate", "global_gradient_norm"]
        else:
            for summ in summaries:
                if summ not in OPTIMIZER_SUMMARIES:
                    raise ValueError(
                        "Summaries should be one of [%s], you provided %s." % (", ".join(OPTIMIZER_SUMMARIES), summ)
                    )
        if learning_rate is not None and learning_rate_decay_fn is not None:
            if global_step is None:
                raise ValueError("global_step is required for learning_rate_decay_fn.")
            lr = learning_rate_decay_fn(lr, global_step)
            if "learning_rate" in summaries:
                tf.compat.v1.summary.scalar("learning_rate", lr)

        if isinstance(optimizer, type) and issubclass(optimizer, tf.compat.v1.train.Optimizer):
            if lr is None:
                raise ValueError(
                    "Learning rate is None, but should be specified if " "optimizer is class (%s)." % optimizer
                )
            opt = optimizer(learning_rate=lr)
        elif isinstance(optimizer, tf.compat.v1.train.Optimizer):
            opt = optimizer
        elif callable(optimizer):
            if learning_rate is not None:
                opt = optimizer(lr)
            else:
                opt = optimizer()
            if not isinstance(opt, tf.compat.v1.train.Optimizer):
                raise ValueError(
                    "Unrecognized optimizer: function should return " "subclass of Optimizer. Got %s." % str(opt)
                )
        else:
            raise ValueError(
                "Unrecognized optimizer: should be string, "
                "subclass of Optimizer, instance of "
                "subclass of Optimizer or function with one argument. "
                "Got %s." % str(optimizer)
            )

        # All trainable variables, if specific variables are not specified.
        if variables is None:
            variables = tf.compat.v1.trainable_variables()

        # Compute gradients.
        gradients = opt.compute_gradients(loss, variables, colocate_gradients_with_ops=colocate_gradients_with_ops)

        # Multiply some gradients.
        if gradient_multipliers is not None:
            gradients = _multiply_gradients(gradients, gradient_multipliers)
            if not gradients:
                raise ValueError(
                    "Empty list of (gradient, var) pairs encountered. This is most "
                    "likely to be caused by an improper value of gradient_multipliers."
                )

        if "global_gradient_norm" in summaries or "gradient_norm" in summaries:
            tf.compat.v1.summary.scalar("global_norm/gradient_norm", tf.linalg.global_norm(list(zip(*gradients))[0]))

        # Optionally clip gradients by global norm.
        if isinstance(clip_gradients, float):
            gradients = _clip_gradients_by_norm(gradients, clip_gradients)
        elif callable(clip_gradients):
            gradients = clip_gradients(gradients)
        elif clip_gradients is not None:
            raise ValueError("Unknown type %s for clip_gradients" % type(clip_gradients))

        # Add scalar summary for loss.
        if "loss" in summaries:
            tf.compat.v1.summary.scalar("loss", loss)

        # Add histograms for variables, gradients and gradient norms.
        for gradient, variable in gradients:
            if isinstance(gradient, tf.IndexedSlices):
                grad_values = gradient.values
            else:
                grad_values = gradient

            if grad_values is not None:
                var_name = variable.name.replace(":", "_")
                if "gradients" in summaries:
                    tf.compat.v1.summary.histogram("gradients/%s" % var_name, grad_values)
                if "gradient_norm" in summaries:
                    tf.compat.v1.summary.scalar("gradient_norm/%s" % var_name, tf.linalg.global_norm([grad_values]))

        if clip_gradients is not None and ("global_gradient_norm" in summaries or "gradient_norm" in summaries):
            tf.compat.v1.summary.scalar(
                "global_norm/clipped_gradient_norm", tf.linalg.global_norm(list(zip(*gradients))[0])
            )

        # Create gradient updates.
        grad_updates = opt.apply_gradients(
            gradients, global_step=global_step if increment_global_step else None, name="train"
        )

        # Ensure the train_tensor computes grad_updates.
        # train_tensor = tf.compat.v1.with_dependencies([grad_updates], loss)
        with tf.control_dependencies([grad_updates]):
            train_tensor = tf.identity(loss)

        return train_tensor


def _clip_gradients_by_norm(grads_and_vars, clip_gradients):
    """Clips gradients by global norm."""
    gradients, variables = zip(*grads_and_vars)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_gradients)
    return list(zip(clipped_gradients, variables))


def _multiply_gradients(grads_and_vars, gradient_multipliers):
    """Multiply specified gradients."""
    multiplied_grads_and_vars = []
    for grad, var in grads_and_vars:
        if grad is not None and (var in gradient_multipliers or var.name in gradient_multipliers):
            key = var if var in gradient_multipliers else var.name
            multiplier = gradient_multipliers[key]
            if isinstance(grad, tf.IndexedSlices):
                grad_values = grad.values * multiplier
                grad = tf.IndexedSlices(grad_values, grad.indices, grad.dense_shape)
            else:
                grad *= tf.cast(multiplier, grad.dtype)
        multiplied_grads_and_vars.append((grad, var))
    return multiplied_grads_and_vars


def optimizer(loss_fn, **kwargs):

    global_step = tf.compat.v1.train.get_or_create_global_step()
    clip = kwargs.get("clip", None)
    optim = kwargs.get("optim", "sgd")
    eta = kwargs.get("lr", kwargs.get("eta", 0.01))
    lr_scheduler = create_lr_scheduler(**kwargs)
    colocate_gradients_with_ops = bool(kwargs.get("colocate_gradients_with_ops", False))
    sgd_mom = float(kwargs.get("mom", 0.9))
    if optim == "adadelta":
        rho = float(kwargs.get("rho", 0.95))
        eps = float(kwargs.get("epsilon", 1e-6))
        logger.info("adadelta(eta=%f, rho=%f, epsilon=%f)", eta, rho, eps)
        optz = lambda lr: tf.compat.v1.train.AdadeltaOptimizer(lr, rho, eps)
    elif optim == "adam":
        beta1 = float(kwargs.get("beta1", 0.9))
        beta2 = float(kwargs.get("beta2", 0.999))
        eps = float(kwargs.get("epsilon", 1e-8))
        logger.info("adam(eta=%f beta1=%f, beta2=%f, eps=%f)", eta, beta1, beta2, eps)
        optz = lambda lr: tf.compat.v1.train.AdamOptimizer(lr, beta1, beta2, eps)
    elif optim == "adamw":
        wd = float(kwargs.get("weight_decay", 0))
        beta1 = float(kwargs.get("beta1", 0.9))
        beta2 = float(kwargs.get("beta2", 0.999))
        eps = float(kwargs.get("epsilon", 1e-8))
        logger.info("adamw(eta=%f beta1=%f, beta2=%f, eps=%f, wd=%f)", eta, beta1, beta2, eps, wd)
        optz = lambda lr: AdamWOptimizer(lr, wd, beta1, beta2, eps)
    elif optim == "rmsprop":
        # Get mom again with difference default
        mom = float(kwargs.get("mom", 0.0))
        logger.info("rmsprop(eta=%f, mom=%f)", eta, mom)
        optz = lambda lr: tf.compat.v1.train.RMSPropOptimizer(lr, momentum=mom)
    elif sgd_mom > 0:
        logger.info("sgd-mom(eta=%f, mom=%f)", eta, sgd_mom)
        optz = lambda lr: tf.compat.v1.train.MomentumOptimizer(lr, sgd_mom)
    else:
        logger.info("sgd(eta=%f)", eta)
        optz = lambda lr: tf.compat.v1.train.GradientDescentOptimizer(lr)

    logger.info("clip gradients at %s", clip)
    return (
        global_step,
        _optimize_loss(
            loss_fn,
            global_step,
            eta,
            optz,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            clip_gradients=clip,
            learning_rate_decay_fn=lr_scheduler,
            increment_global_step=True,
            variables=kwargs.get('variables')
        ),
    )


class EagerOptimizer:
    def __init__(self, loss, optimizer=None, **kwargs):
        self.loss = loss
        if "lr_function" in kwargs:
            lr_function = kwargs["lr_function"]
        else:
            if "lr_scheduler_type" not in kwargs:
                kwargs["lr_scheduler_type"] = "default"
            lr_function = create_lr_scheduler(**kwargs)
        # decay_fn = None
        # Right now this option is pointless since sparse updates dont work on GPU.  We just turn it off
        sgd_mom = float(kwargs.get("mom", 0.9))
        self.clip = kwargs.get("clip", 100)

        if optimizer:
            self.optimizer = optimizer
        else:
            optim = kwargs.get("optim", "sgd")
            lr = kwargs.get("lr", kwargs.get("eta", 0.01))

            if optim == "adadelta":
                rho = float(kwargs.get("rho", 0.95))
                eps = float(kwargs.get("epsilon", 1e-6))
                logger.info("adadelta(eta=%f, rho=%f, epsilon=%f)", lr, rho, eps)
                self.optimizer = tf.keras.optimizers.Adadelta(lr_function, rho, eps)
            elif optim == "adam":
                beta1 = float(kwargs.get("beta1", 0.9))
                beta2 = float(kwargs.get("beta2", 0.999))
                eps = float(kwargs.get("epsilon", 1e-8))
                logger.info("adam(eta=%f beta1=%f, beta2=%f, eps=%f)", lr, beta1, beta2, eps)
                self.optimizer = tf.keras.optimizers.Adam(lr_function, beta1, beta2, eps)

            elif optim == "adamw":
                import tensorflow_addons as tfa
                beta1 = float(kwargs.get("beta1", 0.9))
                beta2 = float(kwargs.get("beta2", 0.999))
                eps = float(kwargs.get("epsilon", 1e-8))
                wd = float(kwargs.get("weight_decay", 0.0))
                if wd == 0.0:
                    logger.info("adam(eta=%f beta1=%f, beta2=%f, eps=%f)", lr, beta1, beta2, eps)
                    self.optimizer = tf.keras.optimizers.Adam(lr_function, beta1, beta2, eps)
                else:
                    def weight_decay_fn():
                        wd_t = lr_function(tf.cast(self.global_step, tf.float32) / lr) * wd
                        return wd_t
                    logger.info("adamw(eta=%f beta1=%f, beta2=%f, eps=%f, wd=%f)", lr, beta1, beta2, eps, wd)
                    self.optimizer = tfa.optimizers.AdamW(
                        weight_decay=weight_decay_fn, learning_rate=lr_function, beta_1=beta1, beta_2=beta2, epsilon=eps
                    )
            elif optim == "rmsprop":
                # Get mom again with difference default
                mom = float(kwargs.get("mom", 0.0))
                logger.info("rmsprop(eta=%f, mom=%f)", lr, mom)
                self.optimizer = tf.keras.optimizers.RMSprop(lr_function, momentum=mom)
            elif sgd_mom > 0:
                logger.info("sgd-mom(eta=%f, mom=%f)", lr, sgd_mom)
                self.optimizer = tf.keras.optimizers.SGD(lr_function, sgd_mom)
            else:
                logger.info("sgd(eta=%f)", lr)
                self.optimizer = tf.keras.optimizers.SGD(lr_function)

        logger.info("clip gradients at %s", self.clip)

    @property
    def global_step(self):
        return self.optimizer.iterations

    def update(self, model, x, y, num_replicas=1):
        with tf.GradientTape() as tape:
            loss_value = self.loss(model, x, y) / num_replicas
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.clip)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_value

    def update_with_hidden(self, model, h, x, y):
        with tf.GradientTape() as tape:
            loss_value, h = self.loss(model, h, x, y)

        grads = tape.gradient(loss_value, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.clip)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_value, h


class OptimizerManager:
    def __init__(self, model, global_step=0, **kwargs):
        self.model = model
        self.global_step = global_step
        if "lr_function" in kwargs:
            self.lr_function = kwargs["lr_function"]
        else:
            if "lr_scheduler_type" not in kwargs:
                kwargs["lr_scheduler_type"] = "default"
            self.lr_function = create_lr_scheduler(**kwargs)
        self._init_optimizer(**kwargs)
        self.current_lr = 0

    @property
    def global_step(self):
        return self._global_step

    @global_step.setter
    def global_step(self, value):
        self._global_step = value

    def _init_optimizer(self, **kwargs):
        lr_function = self.lr_function
        optim = kwargs.get("optim", "sgd")
        lr = kwargs.get("lr", kwargs.get("eta", 0.01))
        sgd_mom = float(kwargs.get("mom", 0.9))
        if optim == "adadelta":
            rho = float(kwargs.get("rho", 0.95))
            eps = float(kwargs.get("epsilon", 1e-6))
            logger.info("adadelta(eta=%f, rho=%f, epsilon=%f)", lr, rho, eps)
            self.optimizer = tf.keras.optimizers.Adadelta(lr_function, rho, eps)
        elif optim == "adam":
            beta1 = float(kwargs.get("beta1", 0.9))
            beta2 = float(kwargs.get("beta2", 0.999))
            eps = float(kwargs.get("epsilon", 1e-8))
            logger.info("adam(eta=%f beta1=%f, beta2=%f, eps=%f)", lr, beta1, beta2, eps)
            self.optimizer = tf.keras.optimizers.Adam(lr_function, beta1, beta2, eps)

        elif optim == "adamw":
            import tensorflow_addons as tfa
            beta1 = float(kwargs.get("beta1", 0.9))
            beta2 = float(kwargs.get("beta2", 0.999))
            eps = float(kwargs.get("epsilon", 1e-8))
            wd = float(kwargs.get("weight_decay", 0.0))
            if wd == 0.0:
                logger.info("adam(eta=%f beta1=%f, beta2=%f, eps=%f)", lr, beta1, beta2, eps)
                self.optimizer = tf.keras.optimizers.Adam(lr_function, beta1, beta2, eps)
            else:
                def weight_decay_fn():
                    wd_t = lr_function(tf.cast(self.global_step, tf.float32) / lr) * wd
                    return wd_t
                logger.info("adamw(eta=%f beta1=%f, beta2=%f, eps=%f, wd=%f)", lr, beta1, beta2, eps, wd)
                self.optimizer = tfa.optimizers.AdamW(
                    weight_decay=weight_decay_fn, learning_rate=lr_function, beta_1=beta1, beta_2=beta2, epsilon=eps
                )
        elif optim == "rmsprop":
            # Get mom again with difference default
            mom = float(kwargs.get("mom", 0.0))
            logger.info("rmsprop(eta=%f, mom=%f)", lr, mom)
            self.optimizer = tf.keras.optimizers.RMSprop(lr_function, momentum=mom)
        elif sgd_mom > 0:
            logger.info("sgd-mom(eta=%f, mom=%f)", lr, sgd_mom)
            self.optimizer = tf.keras.optimizers.SGD(lr_function, sgd_mom)
        else:
            logger.info("sgd(eta=%f)", lr)
            self.optimizer = tf.keras.optimizers.SGD(lr_function)

    def _identity(self, _):
        return self.current_lr

    def step(self, tape, loss_value):
        """Runs at every step and updates the learning rate

        :return:
        """
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        # TODO grad accum?
        # TODO, add me back
        #grads, _ = tf.clip_by_global_norm(grads, self.clip)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.global_step += 1

