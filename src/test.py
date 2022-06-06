import torch
import torch.nn as nn
import time
from binarized_modules import Binarize, BinarizeConv2d, BinarizeLinear
torch.set_grad_enabled(False)


class Net(nn.Module):
    def __init__(self, channels=100):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, channels)
        self.htanh1 = nn.Tanh()
        self.fc2 = BinarizeLinear(channels, channels)
        self.htanh2 = nn.Tanh()
        self.fc3 = nn.Linear(channels, 10)
        self.logsoftmax=nn.LogSoftmax()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        return self.logsoftmax(x)

class NetBin(nn.Module):
    def __init__(self, channels=100):
        super(NetBin, self).__init__()
        self.fc1 = BinarizeLinear(784, channels)
        self.htanh1 = nn.Hardtanh()
        self.fc2 = BinarizeLinear(channels, channels)
        self.htanh2 = nn.Hardtanh()
        self.fc3 = nn.Linear(channels, 10)
        self.logsoftmax=nn.LogSoftmax()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        return self.logsoftmax(x)


nnbin = NetBin()
nn = Net()
nnbin.eval()
nn.eval()

x = torch.ones((32768, 28, 28), dtype=torch.float32)

t = time.time()
nnbin(x)
print('With Binarization:', time.time() - t)

t = time.time()
nn(x)
print('Without Binarization:', time.time() - t)
