import torch
from torch import nn
from ptflops import get_model_complexity_info

# Example model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Instantiate model
model = SimpleModel()

# Calculate FLOPs and parameters
macs, params = get_model_complexity_info(model, (3, 8, 8), as_strings=True, print_per_layer_stat=True)
print(f'FLOPs: {macs}, Parameters: {params}')
