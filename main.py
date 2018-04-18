import os

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from model import Model
import parser

model = Model()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

save_file = "saves/model.state"
if os.path.isfile(save_file):
    state = torch.load(save_file)
    model.load_state_dict(state)

def train():
    full_data = np.array(parser.get_data("data/sign_mnist_train.csv"), dtype=object)
    while True:
        for i in range(5000):
            sub_data = full_data[np.random.randint(0, len(full_data), 64)]
            data, target = np.stack(sub_data[:, 1]), np.array(sub_data[:, 0], dtype=np.int32)
            data = np.expand_dims(data, axis=1)
            data, target = Variable(torch.Tensor(data)), Variable(torch.LongTensor(target))
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        print(loss.data[0])
        torch.save(model.state_dict(), save_file)

if __name__ == "__main__":
    train()
