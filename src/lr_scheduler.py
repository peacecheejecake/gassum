import logging
from torch.optim.lr_scheduler import _LRScheduler
import math


class LRSchedulerBase:
    def __init__(self, optimizer, max_lr):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self._last_lr = max_lr
        self.init_lr()
        self._step_count = 0

    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            if param_group.get('lr') is None:
                param_group['lr'] = self._last_lr

    def get_lr(self, step):
        raise NotImplementedError

    def get_last_lr(self):
        return self._last_lr
    
    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def step(self):
        self._last_lr = self.get_lr()
        self._step_count += 1
        self.set_lr(self.get_lr())


class ConstantLRScheduler(LRSchedulerBase):
    """ 
        Dummy constant lr scheduler.
    """

    def __init__(self, optimizer, lr, warmup_steps, warmup_start_lr=0.):
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr

        super().__init__(optimizer, lr)
        
    def get_lr(self, step=None):
        if step is None:
            step = self._step_count

        if step < self.warmup_steps:
            lr = self.warmup_start_lr + (self.max_lr - self.warmup_start_lr) * (self.warmup_steps - step)
        else:
            lr = self.max_lr
        return lr


class CosineDecayLRScheduler(LRSchedulerBase):

    def __init__(
        self, 
        optimizer, 
        max_lr, 
        alpha,
        warmup_steps,
        total_steps,
        warmup_start_lr=0.,
        constant_steps=0,
    ):
        if warmup_steps > 0 and warmup_steps + constant_steps > total_steps:
            logging.warn(
                "warmup_steps + constant_steps > total_steps. "
                "At the end of warmup, learning rate would be incontinous."
            )
        
        self.warmup_start_lr = warmup_start_lr
        self.warmup_steps = warmup_steps
        self.decay_start = warmup_steps + constant_steps
        self.decay_steps = total_steps - self.decay_start
        self.alpha = alpha

        super().__init__(optimizer, max_lr)
        
    def get_lr(self, step=None):
        if step is None:
            step = self._step_count

        if step < self.warmup_steps:
            lr = (
                self.warmup_start_lr + 
                (self.max_lr - self.warmup_start_lr) / self.warmup_steps * step
            )
        elif step >= self.decay_start and step < self.decay_start + self.decay_steps:
            _x = (step - self.decay_start) / self.decay_steps * math.pi
            decay = self.alpha + (1 - self.alpha) * 0.5 * (1 + math.cos(_x))
            lr = decay * self.max_lr
        else:
            lr = self.max_lr
        return lr
       

class CosineAnnealingRestartLRScheduler(LRSchedulerBase):

    def __init__(
        self, 
        optimizer, 
        max_lr, 
        alpha, 
        warmup_steps, 
        restart_steps, 
        decay_steps, 
        total_steps, 
        max_decay_mode='exp', 
        gamma=1., 
        warmup_start_lr=0.,
        constant_steps=0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.constant_steps = constant_steps
        self.restart_steps = restart_steps
        self.decay_steps = decay_steps
        self.first_cycle_start = warmup_steps + constant_steps
        self.cycle = decay_steps + restart_steps
        self.total_steps = total_steps
        self.initial_max_lr = max_lr
        self.alpha = alpha
        self.max_decay_mode = max_decay_mode
        self.gamma = gamma

        super().__init__(optimizer, max_lr)

    def get_lr(self, step=None):
        if step is None:
            step = self._step_count

        if step < self.warmup_steps:
            lr = self.warmup_start_lr + (self.max_lr - self.warmup_start_lr) / self.warmup_steps * step
        elif step >= self.first_cycle_start:
            second_cycle_start = self.first_cycle_start + self.decay_steps
            if step < second_cycle_start:
                _x = (step - self.first_cycle_start) % self.decay_steps / self.decay_steps * math.pi
                decay = self.alpha + (1 - self.alpha) * 0.5 * (1 + math.cos(_x))
                lr = decay * self.max_lr
            else:
                step_in_cycle = (step - second_cycle_start) % self.cycle
                cycle_max_lr = self.get_cycle_max_lr(step)
                cycle_min_lr = self.alpha * self.max_lr
                cycle_alpha = cycle_min_lr / cycle_max_lr
                if step_in_cycle < self.restart_steps:
                    lr = cycle_min_lr + (cycle_max_lr - cycle_min_lr) / self.restart_steps * step_in_cycle
                else:
                    _x = (step_in_cycle - self.restart_steps) / self.decay_steps * math.pi
                    decay = cycle_alpha + (1 - cycle_alpha) * 0.5 * (1 + math.cos(_x))
                    lr = decay * cycle_max_lr
        else:
            lr = self.max_lr
        return lr

    def get_cycle_max_lr(self, step=None):
        second_cycle_start = self.first_cycle_start + self.decay_steps
        if self.max_decay_mode == 'cos':
            cycle_start = (
                0 
                if step < second_cycle_start 
                else second_cycle_start + (step - second_cycle_start) // self.cycle * self.cycle
            )
            _x = cycle_start / (self.total_steps - self.first_cycle_start) * math.pi / 1.5
            decay = self.alpha + (1 - self.alpha) * 0.5 * (1 + math.cos(_x))
        elif self.max_decay_mode == 'exp':
            decay = (
                1 
                if step < second_cycle_start
                else self.gamma ** ((step - second_cycle_start) // self.cycle + 1)
            )
        else:
            raise ValueError('max_decay_mode must be either "cos" or "exp".')
        return decay * self.max_lr


class SimpleLRScheduler(LRSchedulerBase):

    def __init__(self, optimizer, lr, warmup_steps):
        self.warmup_steps = warmup_steps

        super().__init__(optimizer, lr)
        
    def get_lr(self, step=None):
        if step is None:
            step = self._step_count
        return self.max_lr * min(step ** -0.5, step * self.warmup_steps ** -1.5)
        