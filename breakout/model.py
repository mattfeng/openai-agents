import torch as T
import torch.nn as nn
import torch.functional

class DQN(nn.Module):
    """
    DQN (Deep Q-Network)

    """

    def __init__(self):
        super(DQN, self).__init__()
        # input size (210, 160, 3)
        self.conv1 = nn.Conv2d(3, 32)
        self.bn1 = nn.BatchNorm2d(32) # num_features = # channels
        self.conv2 = nn.Conv2d(32, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()

    def forward(self, x):
        """

        Parameters
        ----------
        x :


        """
        pass
        