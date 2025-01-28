from HFRPCModel.VehicleDynamics import vehicle_dynamics
import torch
import torch.nn as nn


class VehicleDynamicsNetwork(nn.Module):
    def __init__(self, params, dt):
        super(VehicleDynamicsNetwork, self).__init__()
        self.params = params
        self.dt = dt

    def forward(self, current_state, control_input):
        """
        Compute the next state based on the current state and control input.

        Args:
        - current_state: Tensor of shape [batch_size, state_dim], the current state.
        - control_input: Tensor of shape [batch_size, control_dim], the control input.

        Returns:
        - next_state: Tensor of shape [batch_size, state_dim], the next state.
        """
        next_state = vehicle_dynamics(current_state, control_input, self.params, self.dt)
        return next_state
