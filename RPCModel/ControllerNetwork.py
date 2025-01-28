import torch
import torch.nn as nn


# 控制策略神经网络
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
            # nn.Linear(in_features=self.num_state + self.num_reference - 1, out_features=256),
            nn.Linear(in_features=self.num_state + self.num_reference, out_features=256), ##########################################################
            nn.GELU(),
            nn.Linear(in_features=256, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=self.num_control),  # 单步控制量输出
        )

    def forward(self, features):
        # 分割输入为初始状态和参考值
        # features[:, 0] = 0
        initial_state = features[:, :self.num_state]
        # initial_state[:, 0] = 0
        reference = features[:, self.num_state:self.num_state + self.num_reference]
        # features[:, 1] = features[:, 1] - reference[:, 0]  # 将 features 第二个元素变为原值与 reference 第一个值的差
        # reference[:, 0] = 0  # 将 reference 第一个值设置为 0


        # 网络输出控制量
        outputs = self.fc(features) ##################################################################################################################
        # outputs = self.fc(features[:, 1:self.num_state + self.num_reference])

        # 通过 tanh 激活并映射到实际控制范围
        outputs = torch.tanh(outputs)
        acc = outputs[..., 0:1] * (self.acc_max - self.acc_min) / 2 + (self.acc_max + self.acc_min) / 2
        delta_f = outputs[..., 1:2] * (self.delta_f_max - self.delta_f_min) / 2 + (self.delta_f_max + self.delta_f_min) / 2
        control = torch.cat([acc, delta_f], dim=-1)  # [batch_size, num_control]

        return initial_state, control, reference
