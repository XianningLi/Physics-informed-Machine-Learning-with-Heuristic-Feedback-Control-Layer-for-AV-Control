import torch
import torch.nn as nn


class ControllerNetwork(nn.Module):
    def __init__(self, num_state, num_reference, num_control, acc_min, acc_max, delta_f_min, delta_f_max):
        super(ControllerNetwork, self).__init__()
        self.num_state = num_state
        self.num_reference = num_reference
        self.num_control = num_control
        self.acc_min = acc_min
        self.acc_max = acc_max
        self.delta_f_min = delta_f_min
        self.delta_f_max = delta_f_max

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.num_state + self.num_reference, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=self.num_control * self.num_state),
        )

    def forward(self, features):
        # Split input into initial state and reference values
        initial_state = features[:, :self.num_state]
        reference = features[:, self.num_state:self.num_state + self.num_reference]

        # Construct reference state to match the shape of the initial state
        ref_state = torch.zeros_like(initial_state)
        ref_state[:, 1] = reference[:, 0]  # Lateral position referenceHFRPC_train_GPU.py
        ref_state[:, 3] = reference[:, 1]  # Speed reference

        # Compute state error
        state_error = initial_state - ref_state

        # Compute output gain matrix
        outputs = self.fc(features)
        gain_matrix = outputs.view(-1, self.num_control, self.num_state)

        # Adjust gain matrix values for specific states
        gain_matrix[:, :, 0] = 0  # Zero gain for the first state
        gain_matrix[:, 1, 1] = -gain_matrix[:, 1, 1].clone() ** 2  # Negative square for the second state
        gain_matrix[:, 0, 3] = -gain_matrix[:, 0, 3].clone() ** 2 - 0.6  # Negative square for the fourth state
        gain_matrix[:, 0, 1] = 0  # Negative square for the second state
        gain_matrix[:, 1, 3] = 0  # Negative square for the fourth state

        # Compute control: u = K * (x - x_ref)
        control = torch.bmm(gain_matrix, state_error.unsqueeze(-1)).squeeze(-1)

        # Scale control values to the valid range
        control = torch.tanh(control)
        acc = control[..., 0:1] * (self.acc_max - self.acc_min) / 2 + (self.acc_max + self.acc_min) / 2
        delta_f = control[..., 1:2] * (self.delta_f_max - self.delta_f_min) / 2 + (
                self.delta_f_max + self.delta_f_min) / 2
        control = torch.cat([acc, delta_f], dim=-1)

        return initial_state, control, reference
