import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class QuoridorNNet(nn.Module):
    def __init__(self, game, args):
        super(QuoridorNNet, self).__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        self.conv_in = nn.Conv2d(4, args.num_channels, kernel_size=3, stride=1, padding=1)
        self.bn_in = nn.BatchNorm2d(args.num_channels)

        self.res_blocks = nn.ModuleList([ResidualBlock(args.num_channels) for _ in range(args.num_res_blocks)])

        # Value Head
        self.conv_v = nn.Conv2d(args.num_channels, 1, kernel_size=1)  # 1x1 conv
        self.bn_v = nn.BatchNorm2d(1)
        self.fc_v1 = nn.Linear(self.board_x * self.board_y, 256)
        self.fc_v2 = nn.Linear(256, 1)

        # Policy Head
        self.conv_pi = nn.Conv2d(args.num_channels, 2, kernel_size=1)  # 1x1 conv
        self.bn_pi = nn.BatchNorm2d(2)
        self.fc_pi = nn.Linear(2 * self.board_x * self.board_y, self.action_size)

    def forward(self, s):
        s = s.view(-1, 4, self.board_x, self.board_y)
        s = F.relu(self.bn_in(self.conv_in(s)))

        for block in self.res_blocks:
            s = block(s)

        # Value Head
        v = F.relu(self.bn_v(self.conv_v(s)))
        v = v.view(-1, self.board_x * self.board_y)
        v = F.relu(self.fc_v1(v))
        v = self.fc_v2(v)

        # Policy Head
        pi = F.relu(self.bn_pi(self.conv_pi(s)))
        pi = pi.view(-1, 2 * self.board_x * self.board_y)
        pi = self.fc_pi(pi)

        return F.log_softmax(pi, dim=1), torch.tanh(v)
