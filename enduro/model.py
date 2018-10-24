import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DuelingDQN(nn.Module):
    def __init__(self, in_channels, num_moves):
        super(DuelingDQN, self).__init__()
        self.num_moves = num_moves
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.adv_fc1 = nn.Linear(7 * 7 * 64, 512)
        self.adv_fc2 = nn.Linear(512, num_moves)

        self.val_fc1 = nn.Linear(7 * 7 * 64, 512)
        self.val_fc2 = nn.Linear(512, 1)
    
    def forward(self, x):
        x = x.view(-1, 2, 84, 84)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, 7 * 7 * 64)

        adv = F.relu(self.adv_fc1(x))
        adv = F.relu(self.adv_fc2(adv))

        val = F.relu(self.val_fc1(x))
        val = F.relu(self.val_fc2(val)).expand(-1, self.num_moves)

        print("adv", adv - adv.mean(1, keepdim=True).expand(-1, self.num_moves))
        print("val", val)

        x = val + (adv - adv.mean(1, keepdim=True).expand(-1, self.num_moves))

        return x

if __name__ == "__main__":
    net = DuelingDQN(2, 9)
    i = T.from_numpy(np.zeros(84 * 84 * 2).reshape((2, 84, 84))).float()
    print(net(i))
