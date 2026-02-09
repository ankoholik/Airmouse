import torch
import torch.nn as nn

from airmouse.gestures.registry import GESTURES


class GestureModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, with_softmax=False):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.fc4(out)
        if with_softmax:
            out = self.softmax(out)
        return out


input_size = 63  # 21 landmarks × 3 coordinates
hidden_size = 128
num_classes = len(GESTURES)

model = GestureModel(input_size, hidden_size, num_classes)
