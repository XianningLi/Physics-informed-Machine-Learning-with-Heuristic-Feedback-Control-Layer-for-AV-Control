from DPCModel.VehicleDynamics import vehicle_dynamics
import torch
import torch.nn as nn


class VehicleDynamicsNetwork(nn.Module):
    def __init__(self, params, dt, prediction_horizon):
        super(VehicleDynamicsNetwork, self).__init__()
        self.params = params
        self.dt = dt
        self.prediction_horizon = prediction_horizon

    def forward(self, initial_state, control_sequence, reference):
        # Assuming control_sequence has shape [batch_size, prediction_horizon, control_dim]
        # and initial_state has shape [batch_size, state_dim]
        # and reference is some tensor you want to return unchanged
        states = [initial_state]
        for i in range(self.prediction_horizon):
            next_state = vehicle_dynamics(states[-1], control_sequence[:, i, :], self.params, self.dt)
            states.append(next_state)
        # Return all states for the prediction horizon, along with the initial state and reference
        return torch.stack(states, dim=1), initial_state, reference, control_sequence
