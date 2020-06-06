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
from gradcam import GradCAM, visualize_cam
from plot import Plot
from torchvision.utils import make_grid, save_image
import numpy as np
import matplotlib.pyplot as plt


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

    def get_gradcam_images(self, model,layers,image_list,classes, figsize = (23,33),sub_plot_rows = 9, sub_plot_cols = 3, image_count=25 ):
      fig = plt.figure(figsize=figsize)
      for i in range(image_count):
        heat_map_image = [image_list[i][0].cpu()/2+0.5]
        result_image = [image_list[i][0].cpu()/2+0.5]
        for model_layer in layers:
          grad_cam = GradCAM(model, model_layer)
          mask, _ = grad_cam(image_list[i][0].clone().unsqueeze_(0))
          heatmap, result = visualize_cam(mask, image_list[i][0].clone().unsqueeze_(0)/2+0.5)
          heat_map_image.extend([heatmap])
          result_image.extend([result])
        grid_image = make_grid(heat_map_image+result_image,nrow=len(layers)+1,pad_value=1)
        npimg = grid_image.numpy()
        sub = fig.add_subplot(sub_plot_rows, sub_plot_cols, i+1)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        sub.set_title('P = '+classes[int(image_list[i][1])]+" A = "+classes[int(image_list[i][2])],fontweight="bold",fontsize=18)
        sub.axis("off")
        plt.tight_layout()
        fig.subplots_adjust(wspace=0)
    
        
    @staticmethod
    def change(pil_img,device):
      torch_img = transforms.Compose([
          transforms.Resize((32, 32)),
          transforms.ToTensor()
      ])(pil_img).to(device)
      normed_torch_img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(torch_img)[None]
      return torch_img,normed_torch_img    


    @staticmethod
    def apply_l1_regularizer(model, loss, l1_factor=0.0005):
        reg_loss = 0
        parameters = model.parameters()
        for param in parameters:
          reg_loss += torch.sum(param.abs())
        loss += l1_factor * reg_loss
        return loss





