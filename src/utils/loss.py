import torch
import torch.nn as nn

class InternalLossWeighter(nn.Module):
    def __init__(self, num_losses=3):
        super().__init__()
        # 初始化为0，即初始权重为 e^0 = 1
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses):
        weighted_losses = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_losses.append(precision * loss + self.log_vars[i])
        return sum(weighted_losses)