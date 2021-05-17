from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Nu(nn.Module):
    """
    Class for Lagrangian multiplier.

    :param penalty_init: The value with which to initialize the Lagrange multiplier with
    :param clamp_at    : The minimum value to allow nu to drop to. If None, is set to penalty_init
    """
    def __init__(self, penalty_init=1., clamp_at=None):
        super(Nu, self).__init__()
        self.penalty_init = penalty_init
        penalty_init = np.log(max(np.exp(penalty_init)-1, 1e-8))
        self.log_nu = nn.Parameter(penalty_init*torch.ones(1))
        self.clamp_at = penalty_init if clamp_at is None else clamp_at

    def forward(self):
        #return self.log_nu.exp()
        return F.softplus(self.log_nu)

    def clamp(self):
        self.log_nu.data.clamp_(
                min=np.log(max(np.exp(self.clamp_at)-1, 1e-8)))


class DualVariable:
    """
    Class for handling the Lagrangian multiplier.

    :param alpha: The budget size
    :param learning_rate: Learning rate for the Lagrange multiplier
    :param penalty_init: The value with which to initialize the Lagrange multiplier with
    """
    def __init__(self, alpha=0, learning_rate=10, penalty_init=1, clamp_at=None):
        self.nu = Nu(penalty_init, clamp_at)
        self.alpha = alpha
        self.loss = torch.tensor(0)
        self.optimizer = optim.Adam(
                self.nu.parameters(), lr=learning_rate)

    def update_parameter(self, cost):
        # Compute loss.
        self.loss = - self.nu() * (cost-self.alpha)

        # Update.
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        # Clamp.
        self.nu.clamp()


class PIDLagrangian:
    """
    Implements PID Lagrangian update.
    Provides similar interface as Lagrangian.
    :param Kp: Kp paramater of PID
    :param Ki: Ki parameter of PID
    :param Kd: Kd parameter of PID
    :param penalty_init: Initial value of penalty.
    :param pid_delay: Memory of PID
    :param delta_p_ema_alpha: Exponential moving average dampening
                              factor for delta_p calculation.
    :param delta_d_ema_alpha: Exponential moving average dampening
                              factor for delta_d calculation.
    """
    def __init__(self, alpha=0, penalty_init=1,
                 Kp=0, Kd=0, Ki=1, pid_delay=10,
                 delta_d_ema_alpha=0.95,
                 delta_p_ema_alpha=0.95):
        self.budget = alpha
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.pid_delay = pid_delay
        self.pid_i = penalty_init
        self.cost_penalty = penalty_init
        self.cost_deltas = deque(maxlen=pid_delay)
        self.cost_deltas.append(0)

        self._delta_p = 0
        self._cost_delta = 0

        # Using smoothed value is akin to how we use momentum
        # in gradient descent
        self.delta_d_ema_alpha = delta_d_ema_alpha
        self.delta_p_ema_alpha = delta_p_ema_alpha

    def update_parameter(self, cost):
        # See https://github.com/astooke/rlpyt/blob/master/rlpyt/projects/safe/cppo_pid.py#L160
        cost = float(cost)
        self.loss = torch.tensor(cost) # for compatibility with dual variable class
        delta = cost - self.budget
        # integral control
        self.pid_i = max(0, self.pid_i + self.Ki*delta)

        # proportional control
        self._delta_p = (self.delta_p_ema_alpha*self._delta_p +
                         (1-self.delta_p_ema_alpha)*delta)
        # derivative control
        self._cost_delta = (self.delta_d_ema_alpha*self._cost_delta +
                            (1 - self.delta_d_ema_alpha)*cost)
        pid_d = max(0, self._cost_delta - self.cost_deltas[0])

        # PID Control
        pid_o = (self.Kp*self._delta_p +
                 self.Kd*pid_d +
                 self.pid_i)
        self.cost_penalty = max(0, pid_o)

        self.cost_deltas.append(self._cost_delta)

    def nu(self):
        # For consistency with DualVariable class.
        return torch.tensor(self.cost_penalty)

