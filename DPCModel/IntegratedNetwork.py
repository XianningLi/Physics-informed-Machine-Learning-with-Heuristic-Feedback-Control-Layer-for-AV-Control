import torch
import torch.nn as nn
from DPCModel.ControllerNetwork import ControllerNetwork  # Assuming already defined
from DPCModel.VehicleDynamicsNetwork import VehicleDynamicsNetwork  # Assuming already defined


class IntegratedNetwork(nn.Module):
    def __init__(self, controller_params, dynamics_params, dt, prediction_horizon):
        super(IntegratedNetwork, self).__init__()
        self.controller = ControllerNetwork(**controller_params)
        self.dynamics = VehicleDynamicsNetwork(dynamics_params, dt, prediction_horizon)

    def forward(self, features):
        initial_state, control_sequence, reference = self.controller(features)
        states, initial_state, reference, control_sequence = self.dynamics(initial_state, control_sequence, reference)
        return states, initial_state, reference, control_sequence
