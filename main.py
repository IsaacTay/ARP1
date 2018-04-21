import os

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from model import Model
import parser

model = Model()

optimizer = optim.Adam(model.parameters(), eps=10**-4)

save_file = "saves/model.state"
if os.path.isfile(save_file):
    state = torch.load(save_file)
    model.load_state_dict(state)
test_accuracy_file = "saves/test_accuracy.state"
test_accuracy = []
if os.path.isfile(test_accuracy_file):
    test_accuracy = torch.load(test_accuracy_file)["test_accuracy"]    
    
train_data = np.array(parser.get_data("data/sign_mnist_train.csv"), dtype=object)
test_data = np.array(parser.get_data("data/sign_mnist_test.csv"), dtype=object)

def train():
    prev_accuracy = 0
    if len(test_accuracy) > 0:
        prev_accuracy = test_accuracy[-1] * 100
    for i in range(1):
        sub_data = train_data[np.random.randint(0, len(train_data), 16)]
        data, target = np.stack(sub_data[:, 1]), np.array(sub_data[:, 0], dtype=np.int32)
        data = np.expand_dims(data, axis=1)
        data, target = Variable(torch.Tensor(data)), Variable(torch.LongTensor(target))
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        print("{} - {}: {}%".format(len(test_accuracy), loss.data[0], prev_accuracy), end='\r')
        loss.backward()
        optimizer.step()

def test():
    global test_accuracy
    data, target = np.stack(test_data[:, 1]), np.array(test_data[:, 0], dtype=np.int32)
    data = np.expand_dims(data, axis=1)
    data = Variable(torch.Tensor(data))
    output = model(data).data.numpy()
    accuracy = np.sum(np.equal(np.argmax(output, axis=1), target))/len(output)
    test_accuracy.append(accuracy)
    
if __name__ == "__main__":
    try:
        while True:
            train()
            test()
    except KeyboardInterrupt:
        torch.save(model.state_dict(), save_file)
        torch.save({"test_accuracy": test_accuracy}, test_accuracy_file)
        pass
