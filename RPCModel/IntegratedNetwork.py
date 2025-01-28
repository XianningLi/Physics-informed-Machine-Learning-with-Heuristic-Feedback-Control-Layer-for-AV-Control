import torch
import torch.nn as nn
from RPCModel.ControllerNetwork import ControllerNetwork
from RPCModel.VehicleDynamicsNetwork import VehicleDynamicsNetwork


class IntegratedNetwork(nn.Module):
    def __init__(self, controller_params, dynamics_params, dt, prediction_horizon):
        super(IntegratedNetwork, self).__init__()
        self.controller = ControllerNetwork(**controller_params)
        self.dynamics = VehicleDynamicsNetwork(dynamics_params, dt)
        self.prediction_horizon = prediction_horizon

    def forward(self, initial_state, reference):
        """
        Perform recursive prediction over the prediction horizon.

        Args:
        - initial_state: Tensor of shape [batch_size, state_dim], the initial state.
        - reference: Tensor of shape [batch_size, reference_dim], the reference input.

        Returns:
        - states: Tensor of shape [batch_size, prediction_horizon+1, state_dim], the predicted states.
        - controls: Tensor of shape [batch_size, prediction_horizon, control_dim], the control inputs.
        """
        batch_size = initial_state.shape[0]

        # Initialize storage for states and controls
        states = [initial_state]
        controls = []

        current_state = initial_state

        for _ in range(self.prediction_horizon):
            # 拼接当前状态和参考值生成 features
            features = torch.cat([current_state, reference], dim=-1)  # [batch_size, state_dim + reference_dim]

            # 控制器生成当前控制量
            _, control, _ = self.controller(features)
            controls.append(control)

            # 动力学模型计算下一状态
            next_state = self.dynamics(current_state, control)
            states.append(next_state)

            # 更新当前状态
            current_state = next_state

        # Stack the results
        states = torch.stack(states, dim=1)  # [batch_size, prediction_horizon+1, state_dim]
        controls = torch.stack(controls, dim=1)  # [batch_size, prediction_horizon, control_dim]

        return states, controls
