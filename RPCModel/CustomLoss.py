import torch
import torch.nn as nn


class CustomLoss(nn.Module):
    def __init__(self, model, Qx_values, Qu_values, Qt_values, device):
        super(CustomLoss, self).__init__()
        self.model = model
        self.Qx = torch.diag(torch.tensor(Qx_values, dtype=torch.float32)).to(device)
        self.Qu = torch.diag(torch.tensor(Qu_values, dtype=torch.float32)).to(device)
        self.Qt = torch.diag(torch.tensor(Qt_values, dtype=torch.float32)).to(device)

    def forward(self, states, controls, reference):
        # 扩展reference到完整状态向量的形式
        ref_extended = torch.zeros_like(states)
        # print(ref_extended)
        batch_size, sequence_length, feature_size = ref_extended.shape
        ref_column = reference[:, 0].unsqueeze(1).expand(-1, sequence_length)  # 扩展以匹配sequence_length
        ref_extended[:, :, 1] = ref_column
        vx_column = reference[:, 1].unsqueeze(1).expand(-1, sequence_length)  # 扩展以匹配sequence_length
        ref_extended[:, :, 3] = vx_column

        # 计算状态损失
        state_diff = states - ref_extended
        state_loss = torch.sum(torch.matmul(state_diff.unsqueeze(2), torch.matmul(self.Qx, state_diff.unsqueeze(-1))), dim=1).squeeze(dim=2)

        # 计算控制损失
        control_loss = torch.sum(torch.matmul(controls.unsqueeze(2), torch.matmul(self.Qu, controls.unsqueeze(-1))), dim=1).squeeze(dim=2)

        # 计算终端状态损失
        terminal_state_diff = states[:, -1, :] - ref_extended[:, -1, :]
        terminal_state_diff = terminal_state_diff.unsqueeze(1)
        terminal_state_loss = torch.sum(torch.matmul(terminal_state_diff.unsqueeze(2), torch.matmul(self.Qt, terminal_state_diff.unsqueeze(-1))), dim=1).squeeze(dim=2)

        # # 计算额外的状态损失差异惩罚
        # # print(state_loss)
        # # state_loss_diff = (state_loss[1:] - 0.9 * state_loss[:-1]) / state_loss[:-1]
        # state_loss_diff = (state_loss[1:] - 0.9 * state_loss[:-1])
        # # print(state_loss_diff.size())
        # # state_loss_diff = state_loss[:, 1:] - 0.5 * state_loss[:, :-1]
        # # print(state_loss_diff)  # / state_loss[:, :-1]
        # state_loss_diff_penalty = torch.where(state_loss_diff > 0, state_loss_diff, torch.zeros_like(state_loss_diff))
        # extra_penalty = torch.sum(state_loss_diff_penalty, dim=1)

        # 总损失
        # loss = state_loss + control_loss + terminal_state_loss + 1 * extra_penalty
        loss = state_loss + control_loss + terminal_state_loss
        lambda2 = 0.2  # L2正则化系数
        l2_regularization = sum(torch.sum(p ** 2) for p in self.model.parameters() if p.requires_grad)

        return torch.mean(loss) #+ lambda2 * l2_regularization
