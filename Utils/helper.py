import torch
from torchvision import datasets, transforms
from google.colab import drive
from torchsummary import summary
import sys
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import OneCycleLR
import torch.optim as optim
import torch.nn.functional as F


class HelperModel(object):

    def __init__(self, model):
        self.model = model

    @staticmethod
    def get_device():
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        return device

    def display_model_summay(self, model, input_image_size):
        summary(model, input_size=input_image_size)

    def get_optimizer(self, lr=0.01, momentum=0.9):
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        return optimizer

    def get_step_lr(self, lr=0.01, momentum=0.9, step_size=1, gamma=0.1, optimizer = None):
       if optimizer is None:
         optimizer = self.get_optimizer(lr, momentum)
       scheduler = StepLR(optimizer, step_size, gamma)
       return scheduler

    def get_one_cycle_lr(self, lr=0.01, momentum=0.9, optimizer = None, max_lr=0.1,total_steps=20):
       if optimizer is None:
         print("no optimser")
         optimizer = self.get_optimizer(lr=lr, momentum = momentum)
       scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps )
       return scheduler


    def get_l2_regularizer(self, weight_decay=0.001, lr=0.01, momentum=0.9):
        l2_regularizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        return l2_regularizer


    @staticmethod
    def apply_l1_regularizer(model, loss, l1_factor=0.0005):
        reg_loss = 0
        parameters = model.parameters()
        for param in parameters:
          reg_loss += torch.sum(param.abs())
        loss += l1_factor * reg_loss
        return loss





