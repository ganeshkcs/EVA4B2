import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from helper import HelperModel
import torch.optim as optim
import matplotlib.pyplot as plt


class LRAccuracy(object):
    def __init__(self):
        self.history = { "train_lr": [], "train_acc": [] }

    def reset_data(self):
      self.history = { "train_lr": [], "train_acc": [] } 

    def get_best_lr(self):
      lr_list = self.history["train_lr"]
      acc_list = self.history["train_acc"]
      best_acc = max(acc_list)
      best_acc_index = acc_list.index(best_acc)
      best_lr = lr_list[best_acc_index]
      print("Best Accuracy", best_acc)
      print("Best Lr", best_lr)
      return best_lr

    def plot_acc_lr(self):
      plt.plot(self.history["train_lr"], self.history["train_acc"])
      plt.ylabel('Train Accuracy')
      plt.xlabel("Learning rate")
      plt.title("Lr v/s Accuracy")
      plt.show()

    def range_test(self, model, device, train_loader, criterion, momentum = 0.9, weight_decay=0.005, l1_factor=None, min_lr = 0.001, max_lr=0.02, epochs = 10, ):
        lr = min_lr
        for epoch in range(epochs):
          optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay) 
          lr += (max_lr - min_lr)/epochs
          model.train()
          pbar = tqdm(train_loader)
          correct = 0
          processed = 0
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

              # Backpropagation
              loss.backward()
              optimizer.step()

              # Update pbar-tqdm

              pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
              correct += pred.eq(target.view_as(pred)).sum().item()
              processed += len(data)

              pbar.set_description(desc=f'epoch = {epoch+1} LR={optimizer.param_groups[0]["lr"]} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
              acc = float("{:.2f}".format(100 * correct / processed))
          self.history["train_acc"].append(100 * correct / processed)
          self.history["train_lr"].append(optimizer.param_groups[0]['lr'])