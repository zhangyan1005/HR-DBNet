import torch
import torch.nn as nn


class SMU(nn.Module):
    def __init__(self, alpha=0.25, mu=1000000):
        super(SMU, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor([alpha]))
        self.mu = nn.Parameter(torch.FloatTensor([mu]))

    def forward(self, x):
        y = (1 + self.alpha) * x + (1 - self.alpha) * x * torch.erf(self.mu * (1 - self.alpha) * x)
        return y / 2.

# ###test###
# sum = SMU()
#
# x = torch.randn([3, 3])
# y = sum(x)
# for parameters in sum.parameters():
#     print(parameters)
#
# print(x)
# print(y)
