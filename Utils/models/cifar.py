import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight=True, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias

class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features * self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features * self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False):  # lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)

#function which handles whether to apply BN or GBN
def norm2d(output_channels, batch_type="BN"):
    if batch_type == "GBN":
        num_splits = 8
        return GhostBatchNorm(output_channels,num_splits)
    else:
        return nn.BatchNorm2d(output_channels)

class Cifar(nn.Module):
    def __init__(self):
        super(Cifar, self).__init__()
        # Input Block

        self.drop_val = 0.1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=1, bias=False), #RF 3
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),

            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3,3), padding=1, bias=False), # RF 5
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),
        )
        self.MP1 = nn.Sequential(
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1,1), bias=False),  # RF 6
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),
        )

        # Convolution Block 2
        self.convblock2=nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1, bias=False), #RF 10
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),

            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3,3), padding=1, bias=False),   # RF 14
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),
        )
        self.MP2 = nn.Sequential(
            nn.MaxPool2d(2,2),  # RF 16
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1,1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),
        )

         # Convolution Block 3
        self.convblock3=nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1,groups = 64, bias=False), #RF 24
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1), padding=0, bias=False), #RF 24
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),


            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=2,dilation=2, bias=False),   # RF 40
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),
        )

        self.MP3 = nn.Sequential(
            nn.MaxPool2d(2,2),  # RF 44
            nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1,1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),
        )

         # Convolution Block 4
        self.convblock4=nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3),groups=32, padding=1, bias=False), #RF 60
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),

            nn.Conv2d(in_channels=32, out_channels=256, kernel_size=(1,1), padding=0, bias=False), #RF 60
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),

            nn.AvgPool2d(kernel_size=4), #RF 84
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1,1),padding=0, bias=False)
        )

    def forward(self,x):
      x = self.convblock1(x)
      x = self.MP1(x)
      x = self.convblock2(x)
      x = self.MP2(x)
      x = self.convblock3(x)
      x = self.MP3(x)
      x = self.convblock4(x)
      x = x.view(-1,10)

      return F.log_softmax(x, dim=-1)

