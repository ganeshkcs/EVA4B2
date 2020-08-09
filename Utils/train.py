import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from helper import HelperModel
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, random_split
import math
from PIL import Image
import cv2
import numpy as np
import torch
import os
from tqdm import notebook
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import albumentations as A
import albumentations.pytorch as AP
import random
from tqdm.notebook import tqdm
from tqdm import tqdm_notebook
import time
import matplotlib.pyplot as plt
from dice import dice_coefficient, iou_score

class Train(object):
    def __init__(self):
        self.train_losses = []
        self.train_acc = []
        self.train_lr = []

    def plot_cycle_lr(self):
      plt.plot(np.arange(1,25), self.train_lr)
      plt.xlabel('Epochs')
      plt.ylabel("Learning rate")
      plt.title("Lr v/s Epochs")
      plt.show()

    def train(self, model, device, train_loader, optimizer, criterion,l1_factor=None,scheduler=None ):
        model.train()
        pbar = tqdm(train_loader)
        correct = 0
        processed = 0
        print('LR:',optimizer.param_groups[0]['lr'])
        self.train_lr.append(optimizer.param_groups[0]['lr'])
        for batch_idx, (data, target) in enumerate(pbar):
            # get samples
            data, target = data.to(device), target.to(device)

            # Init
            optimizer.zero_grad()
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            y_pred = model(data)
            # pdb.set_trace()
            # Calculate loss
            # loss = F.nll_loss(y_pred, target)
            # criterion = nn.CrossEntropyLoss()
            # loss = criterion(y_pred, target)

            loss = criterion(y_pred, target)

            # update l1 regularizer if requested
            if l1_factor:
                loss = HelperModel.apply_l1_regularizer(model, loss, l1_factor)
            self.train_losses.append(loss.item())

            # Backpropagation
            loss.backward()
            optimizer.step()
            if(scheduler):
              scheduler.step()

            # Update pbar-tqdm

            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(desc=f'Train Set: Train Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            acc = float("{:.2f}".format(100 * correct / processed))
            # self.train_acc.append(100*correct/processed)
            self.train_acc.append(acc)
    


    def train_mask_depth(self,model, device, train_loader, optimizer, mask_criterion, depth_criterion, epoch, scheduler = False):
      running_mask_loss = 0
      running_depth_loss=0
      total_loss = 0
      mask_coef = 0
      depth_coef = 0
      model.train()
      pbar = tqdm(train_loader)
      total_length = len(train_loader)
      print('LR:',optimizer.param_groups[0]['lr'])
      self.train_lr.append(optimizer.param_groups[0]['lr'])
      self.train_losses = []
      self.train_acc = []
      acc_mask = 0
      acc_depth = 0
      iou_mask = 0
      iou_dense = 0

      for batch_idx, (data, mask_target, depth_target) in enumerate(pbar):
        # get samples
        data, mask_target, depth_target = data.to(device), mask_target.to(device), depth_target.to(device)

        optimizer.zero_grad()
      
        mask_target = mask_target.unsqueeze_(1)
        depth_target = depth_target.unsqueeze_(1)

        mask_target = torch.sigmoid(mask_target)
        depth_target = torch.sigmoid(depth_target)

        #Predict
        mask_pred, depth_pred = model(data)


        # Calculate loss
        
        mask_loss = mask_criterion(  mask_pred,mask_target,)
        depth_loss = depth_criterion(depth_pred,depth_target)
        loss = mask_loss+ depth_loss
        running_mask_loss += mask_loss.item()
        running_depth_loss += depth_loss.item()

        total_loss += loss.item()

        # mask_coef += dice_coefficient(mask_pred,mask_target, mask= True).item()
        # depth_coef += dice_coefficient(depth_pred, depth_target, mask=False).item()

        iou_mask += iou_score(mask_pred.detach().cpu().numpy(), mask_target.detach().cpu().numpy())
        iou_dense += iou_score(depth_pred.detach().cpu().numpy(), depth_target.detach().cpu().numpy())
        
        # Backpropagation
        loss.backward()
        # torch.autograd.backward([mask_loss, depth_loss])

        optimizer.step()
        if(scheduler):
          scheduler.step()

        pbar.set_description(f'Loss={loss:0.4f}')
        

      # print(f'Mask Coeff ={mask_coef/total_length:0.4f}')
      # print(f'Depth TCoeff ={depth_coef/total_length:0.4f}')

      print(f'IOU Mask={iou_mask/total_length:0.4f}')
      print(f'IOU Depth={iou_dense/total_length:0.4f}')
      
      # train_losses.append((mask_loss/total_length,depth_loss/total_length))
      self.train_losses.append(total_loss/total_length)
      self.train_acc.append((mask_coef + depth_coef)/ total_length)
      # return self.train_losses, self.train_acc