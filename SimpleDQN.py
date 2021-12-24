import torch.nn as nn


class SimpleDQN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(SimpleDQN, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, x):
        return self.pipe(x)
