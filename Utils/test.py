import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from helper import HelperModel
from dice import dice_coefficient, iou_score



class Test(object):

    def __init__(self):
        self.test_losses = []
        self.test_acc = []
        self.misclassified_images = []
        self.trueclassified_images = []

    def update_misclassified_images(self, data, target, pred):
        target_change = target.view_as(pred)
        for i in range(len(pred)):
          if pred[i].item()!= target_change[i].item():
            self.misclassified_images.append([data[i], pred[i], target_change[i]])
          else:
            self.trueclassified_images.append([data[i], pred[i], target_change[i]])

    def test(self, model, device, test_loader, criterion, misclassfied_required=False):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                # criterion = nn.CrossEntropyLoss()
                # test_loss += criterion(output, target).item() 
                
                test_loss += criterion(output, target).item() 
                  
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                # update misclassified images if requested
                if misclassfied_required:
                   self.update_misclassified_images(data, target, pred)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        self.test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        self.test_acc.append(100. * correct / len(test_loader.dataset))
    
    def test_mask_depth(self,model, device, mask_criterion, depth_criterion, test_loader, epoch):
        model.eval()
        mask_loss = 0
        depth_loss = 0
        correct = 0
        mask_coef = 0
        depth_coef = 0
        total_loss = 0
        total_length = len(test_loader)
        self.test_losses = []
        self.test_acc = []
        iou_mask = 0
        iou_dense = 0
        with torch.no_grad():
            for data, mask_target, depth_target in tqdm(test_loader):
                data, mask_target, depth_target = data.to(device), mask_target.to(device), depth_target.to(device)
                mask_target = mask_target.unsqueeze_(1)
                depth_target = depth_target.unsqueeze_(1)
                mask_target = torch.sigmoid(mask_target)
                depth_target = torch.sigmoid(depth_target)
                mask_pred, depth_pred = model(data)
                # mask_loss += mask_criterion(depth_pred,mask_target,).item()  
                # depth_loss += depth_criterion(depth_pred,mask_target,).item()
                mask_loss += mask_criterion(mask_pred,mask_target,).item()  
                depth_loss += depth_criterion(depth_pred,depth_target,).item()
                test_loss = mask_loss+depth_loss
                total_loss += test_loss
                mask_coef += dice_coefficient(mask_pred,mask_target, mask= True).item()
                depth_coef += dice_coefficient(depth_pred, depth_target, mask=False).item()
                iou_mask += iou_score(mask_pred.detach().cpu().numpy(), mask_target.detach().cpu().numpy())
                iou_dense += iou_score(depth_pred.detach().cpu().numpy(), depth_target.detach().cpu().numpy())
    
              
        # test_loss /= (2 * total_length)
        # total_loss /= (2 * total_length)

        test_loss /= (total_length)
        total_loss /= (total_length)

        self.test_losses.append((mask_loss/total_length, depth_loss/total_length,test_loss))
        self.test_acc.append((mask_coef + depth_coef)/ total_length)
        print('\nTest set: Average loss: {:.4f}'.format(test_loss))
        print(f'IOU Mask={iou_mask/total_length:0.4f}')
        print(f'IOU Depth={iou_dense/total_length:0.4f}')
        # print(f'Mask Coefficient ={mask_coef/total_length:0.4f}')
        # print(f'Depth Coefficient ={depth_coef/total_length:0.4f}')
        # return self.test_losses,self.test_acc
