import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import datetime
import pdb
import time
import torchvision.models as torchmodels
from torch.autograd import Variable

class BaseModel(nn.Module):
    def __init__(self, device):
        super(BaseModel, self).__init__()
        if not os.path.exists('logs'):
            os.makedirs('logs')
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S_log.txt')
        self.logFile = open('logs/' + st, 'w')

    def log(self, str):
        # print(str)
        self.logFile.write(str + '\n')

    def criterion(self):
        return nn.CrossEntropyLoss()

    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001)

    def adjust_learning_rate(self, optimizer, epoch, args):
        lr = args.lr * pow(0.9, epoch / 50)  # TODO: Implement decreasing learning rate's rules
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
       


class LazyNet(BaseModel):
    def __init__(self, device):
        super(LazyNet, self).__init__()
        # TODO: Define model here
        self.fc1 = nn.Linear(32 * 32 * 3, 10).to(device)

    def forward(self, x):
        # TODO: Implement forward pass for LazyNet
        x = x.view(-1, 32 * 32 * 3)
        x = self.fc1(x)
        return x
        

class BoringNet(BaseModel):
    def __init__(self, device):
        super(BoringNet, self).__init__()
        # TODO: Define model here
        self.fc1 = nn.Linear(32 * 32 * 3, 120).to(device)
        self.fc2 = nn.Linear(120, 84).to(device)
        self.fc3 = nn.Linear(84, 10).to(device)

    def forward(self, x):
        # TODO: Implement forward pass for BoringNet
        x = x.view(-1, 32 * 32 *3)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class CoolNet(BaseModel):

    def __init__(self, device):
        super(CoolNet, self).__init__(device)

        self.conv1 = nn.Conv2d(3, 64, 5).to(device)
        self.conv2 = nn.Conv2d(64, 16, 5).to(device)

        self.fc1 = nn.Linear(400, 120).to(device)
        self.fc2 = nn.Linear(120, 84).to(device)
        self.fc3 = nn.Linear(84, 10).to(device)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
