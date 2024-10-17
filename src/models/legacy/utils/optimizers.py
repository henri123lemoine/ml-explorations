import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)

np.random.seed(0)


class Optimizer(ABC):
    def __init__(self, lr=0.01, epsilon=1e-7, scheduler=None, **kwargs):
        self.lr = lr
        self.epsilon = epsilon
        self.scheduler = scheduler
        self.global_step = 0

    def update(self, w, dw, vw=None, sw=None):
        adjusted_lr = (
            self.scheduler.get_lr(self.lr, self.global_step)
            if self.scheduler is not None
            else self.lr
        )
        self.global_step += 1
        return self._update(w, dw, vw, sw, adjusted_lr)

    @abstractmethod
    def _update(self, w, dw, vw=None, sw=None, adjusted_lr=None):
        pass

    def decay(self, factor):
        self.lr *= factor


class GD(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _update(self, w, dw, vw=None, sw=None, adjusted_lr=None):
        return w - adjusted_lr * dw, None, None


class Momentum(Optimizer):
    def __init__(self, beta1=0.9, **kwargs):
        super().__init__(**kwargs)
        self.beta1 = beta1

    def _update(self, w, dw, vw=None, sw=None, adjusted_lr=None):
        if vw is None:
            vw = np.zeros_like(dw)
        vw = self.beta1 * vw + (1 - self.beta1) * dw
        return w - adjusted_lr * vw, vw, None


class RMSProp(Optimizer):
    def __init__(self, beta2=0.999, **kwargs):
        super().__init__(**kwargs)
        self.beta2 = beta2

    def _update(self, w, dw, vw=None, sw=None, adjusted_lr=None):
        if sw is None:
            sw = np.zeros_like(dw)
        sw = self.beta2 * sw + (1 - self.beta2) * dw**2
        return w - adjusted_lr * dw / (np.sqrt(sw) + self.epsilon), None, sw


class Adam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, **kwargs):
        super().__init__(**kwargs)
        self.beta1 = beta1
        self.beta2 = beta2

    def _update(self, w, dw, vw=None, sw=None, adjusted_lr=None):
        if vw is None:
            vw = np.zeros_like(dw)
        if sw is None:
            sw = np.zeros_like(dw)
        vw = self.beta1 * vw + (1 - self.beta1) * dw
        sw = self.beta2 * sw + (1 - self.beta2) * dw**2
        corrected_global_step = self.global_step + 1
        vw_corr = vw / (1 - self.beta1**corrected_global_step)
        sw_corr = sw / (1 - self.beta2**corrected_global_step)
        return w - adjusted_lr * vw_corr / (np.sqrt(sw_corr) + self.epsilon), vw, sw