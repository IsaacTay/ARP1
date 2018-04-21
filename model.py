import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 26, 3)
        self.convp1 = nn.Conv2d(26, 26, 1, stride=2)
        self.fc1 = nn.Linear(4394, 260)
        self.fc2 = nn.Linear(260, 26)

    def forward(self, x):
        x = F.relu(self.convp1(self.conv1(x)))
        x = x.view(-1, 4394)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
