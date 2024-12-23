from abc import ABC, abstractmethod

import numpy as np


class LRScheduler(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def get_lr(self, lr, global_step):
        pass


class Constant_(LRScheduler):
    def __init__(self, **kwargs):
        pass

    def get_lr(self, lr, global_step):
        return lr


class ExponentialDecay(LRScheduler):
    def __init__(self, decay_rate, decay_steps, **kwargs):
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def get_lr(self, lr, global_step):
        return lr * (self.decay_rate ** (global_step / self.decay_steps))


class StepDecay(LRScheduler):
    def __init__(self, drop_rate, epochs_drop, **kwargs):
        self.drop_rate = drop_rate
        self.epochs_drop = epochs_drop

    def get_lr(self, lr, global_step):
        return lr * (self.drop_rate ** (global_step // self.epochs_drop))


class TimeBasedDecay(LRScheduler):
    def __init__(self, decay_rate, **kwargs):
        self.decay_rate = decay_rate

    def get_lr(self, lr, global_step):
        return lr / (1 + self.decay_rate * global_step)


class StepWarmup(LRScheduler):
    def __init__(self, warmup_steps, init_lr, **kwargs):
        self.warmup_steps = warmup_steps
        self.init_lr = init_lr

    def get_lr(self, lr, global_step):
        if global_step < self.warmup_steps:
            return self.init_lr + (lr - self.init_lr) * (global_step / self.warmup_steps)
        return lr


class CosineAnnealing(LRScheduler):
    def __init__(self, T_max, eta_min=0, **kwargs):
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self, lr, global_step):
        return (
            self.eta_min + (lr - self.eta_min) * (1 + np.cos(np.pi * global_step / self.T_max)) / 2
        )
