import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class Module(nn.Module):
    def __init__(self, action_space=140):
        super().__init__()

        self.board_conv = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.fence_conv = nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=1)

        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(9 * 9 * 64, action_space)
        self.fc2 = nn.Linear(2, action_space)

        self.flatten = nn.Flatten()
        self.act = nn.ReLU()

    def forward(self, board, fence, num_fences):
        board = self.board_conv(board)
        fence = self.fence_conv(fence)

        x = board + fence

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))

        x = self.fc1(self.flatten(x))
        x = x + self.fc2(num_fences)

        return x

class ResnetModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.board_conv = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        self.fence_conv = nn.Conv2d(1, 3, kernel_size=2, stride=1, padding=1)
        self.fence_fc = nn.Linear(2, 512)
        self.out_fc = nn.Linear(512, 140)

        self.net = resnet18(num_classes=512)

    def forward(self, board, fence, num_fences):
        board = self.board_conv(board)
        fence = self.fence_conv(fence)

        x = board + fence

        x = self.net(x)

        x = x + self.fence_fc(num_fences)
        x = F.relu(x)

        x = self.out_fc(x)

        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = ResnetModule()

    def forward(self, player, board, fence, num_fences):
        board = board.unsqueeze(1)
        fence = fence.unsqueeze(1)

        out = self.net(board, fence, num_fences)

        torch.softmax(out, dim=1)

        return out
