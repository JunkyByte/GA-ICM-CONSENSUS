import torch
import torch.nn as nn
from gym.spaces import Box, Discrete
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self, input_shape, output_size, channels, conv_channels):
        super(Net, self).__init__()
        if len(input_shape.shape) == 1:  # One dimensional observation space
            self.is_image = False
            self.input_shape = input_shape.shape[0]
        elif len(input_shape.shape) == 3:
            self.is_image = True
            self.rescale = True if input_shape.high.any() == 255 else False
            self.input_shape = tuple(input_shape.shape[i] for i in [2, 0, 1])

        if isinstance(output_size, Discrete):
            output_size = output_size.n
            self.is_discrete = True
        elif isinstance(output_size, Box):
            self.out_bounds = torch.tensor([list(output_size.low), list(output_size.high)], dtype=torch.float).to(device)
            output_size = output_size.shape[0]
            self.is_discrete = False

        if self.is_image:  # Conv start
            self.layer1 = nn.Conv2d(self.input_shape[0], conv_channels, (3, 3), stride=2)
            self.act1 = nn.Tanh()
            self.layer2 = nn.Conv2d(conv_channels, conv_channels, (3, 3), stride=2)
            self.act2 = nn.Tanh()
            self.layer2_bis = nn.Conv2d(conv_channels, conv_channels, (3, 3), stride=2)
            self.act2_bis = nn.Tanh()
        else:
            self.layer1 = nn.Linear(self.input_shape, channels)
            self.act1 = nn.Tanh()
            self.layer2 = nn.Linear(channels, channels)
            self.act2 = nn.Tanh()

        if self.is_image:
            size = self.input_shape[1:]
            for i in range(2):
                size = self.conv_output_shape(size, kernel_size=(3, 3), stride=2)
            out_c = self.conv_output_shape(size, kernel_size=(3, 3), stride=2, flat=True) * conv_channels
        else:
            out_c = channels

        self.layer3 = nn.Linear(out_c, channels)
        self.act3 = nn.Tanh()
        self.out_layer = nn.Linear(channels, output_size)

        if not self.is_discrete:
            self.out_act = nn.Sigmoid()

        if self.is_image:
            torch.nn.init.normal_(self.layer1.weight, std=1.)
            torch.nn.init.normal_(self.layer2.weight, std=1.)
            torch.nn.init.normal_(self.layer2_bis.weight, std=1.)
            torch.nn.init.zeros_(self.layer2_bis.bias)
        else:
            torch.nn.init.normal_(self.layer1.weight, std=2.)
            torch.nn.init.normal_(self.layer2.weight, std=2.)

        torch.nn.init.normal_(self.layer3.weight, std=2.)
        torch.nn.init.normal_(self.out_layer.weight, std=2.)
        torch.nn.init.zeros_(self.layer1.bias)
        torch.nn.init.zeros_(self.layer2.bias)
        torch.nn.init.zeros_(self.layer3.bias)
        torch.nn.init.zeros_(self.out_layer.bias)
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x, train=False):
        def pred(x):
            x = x.contiguous()

            if self.is_image:
                x = x.view(-1, *self.input_shape)
                if self.rescale:
                    x = x / 255.
            else:
                x = x.view(-1, self.input_shape)

            x = self.layer1(x)
            x = self.act1(x)
            x = self.layer2(x)
            x = self.act2(x)

            if self.is_image:
                x = self.layer2_bis(x)
                x = self.act2_bis(x)
                x = x.view(x.shape[0], -1)

            x = self.layer3(x)
            x = self.act3(x)
            x = self.out_layer(x)

            if self.is_discrete:
                if train:
                    return x
                else:
                    return torch.argmax(x, axis=1)

            x = self.out_act(x)
            return x * (self.out_bounds[1] - self.out_bounds[0]) + self.out_bounds[0]

        if train:
            hat = pred(x)
        else:
            with torch.no_grad():
                hat = pred(x)
                hat = hat.cpu().detach().numpy()
        return hat

    def conv_output_shape(self, h_w, kernel_size=1, stride=1, pad=0, dilation=1, flat=False):
        from math import floor
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
        w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
        if flat:
            return h * w
        return h, w

