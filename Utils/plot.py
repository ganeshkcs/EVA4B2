import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from google.colab import files

class Plot(object):

    def __init__(self, train_acc, train_losses, test_acc, test_losses):
        self.train_acc = train_acc
        self.train_losses = train_losses
        self.test_acc = test_acc
        self.test_losses = test_losses

    def display_all_plot(self):
        """Plots graph for train, validation accuracy and losses"""
        try:
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))
            axs[0, 0].plot(self.train_losses)
            axs[0, 0].set_title("Training Loss")
            axs[1, 0].plot(self.train_acc)
            axs[1, 0].set_title("Training Accuracy")
            axs[0, 1].plot(self.test_losses)
            axs[0, 1].set_title("Validation Loss")
            axs[1, 1].plot(self.test_acc)
            axs[1, 1].set_title("Validation Accuracy")
        except Exception as err:
            raise err

    def plot_accuracy_graph(self,figsize = (15,10)):
      fig = plt.figure(figsize=figsize)
      ax = plt.subplot()
      accuracy_list = [(self.test_acc,"test_accuracy"),(self.train_acc,"train_accuracy")]
      for data in accuracy_list:
          ax.plot(data[0], label=data[1])
          plt.title("Accuracy Graph")
      ax.legend()
      plt.show()
      
    @staticmethod
    def plot_acc_lr(lr_list = [],acc_list = []):
      plt.plot(lr_list, acc_list)
      plt.ylabel('Train Accuracy')
      plt.xlabel("Learning rate")
      plt.title("Lr v/s Accuracy")
      plt.show()

    @staticmethod
    def plot_cycle_lr(epochs, lr_list):
      plt.plot(np.arange(1,epochs), lr_list)
      plt.xlabel('Epochs')
      plt.ylabel("Learning rate")
      plt.title("Lr v/s Epochs")
      plt.show()
    
    def plot_graph(self, plot_case="Accuracy"):
      fig, ax = plt.subplots()
      if plot_case == "Accuracy":
          train_data = self.train_acc
          test_data = self.test_acc
      else:
          train_data = self.train_losses
          test_data = self.test_losses
      plt.title("Change in training and validation {0}".format(plot_case.lower()))
      plt.xlabel("Num of Epochs")
      plt.ylabel(plot_case)
      ax.plot(train_data, 'r', label='Train')
      ax.plot(test_data, 'b', label='Validation')
      ax.legend(loc='upper right')
      plt.show()

    def plot_train_graph(self, plot_case="Accuracy"):
        try:
            fig = plt.figure(figsize=(9, 9))
            if plot_case == "Accuracy":
                train_data = self.train_acc
            else:
                train_data = self.train_losses
            plt.title("Training {0}".format(plot_case))
            plt.xlabel("Epochs")
            plt.ylabel(plot_case)
            plt.plot(train_data)
            plt.show()
            fig.savefig('train_{0}_graph.png'.format(plot_case.lower()))
        except Exception as err:
            raise err

    def plot_validation_graph(self, plot_case="Accuracy"):
        """Plots single graph for validation losses/accuracy"""
        try:
            fig = plt.figure(figsize=(9, 9))
            if plot_case == "Accuracy":
                test_data = self.test_acc
            else:
                test_data = self.test_losses
            plt.title("Validation {0}".format(plot_case))
            plt.xlabel("Epochs")
            plt.ylabel(plot_case)
            plt.plot(test_data)
            plt.show()
            fig.savefig('validation_{0}_graph.png'.format(plot_case.lower()))
        except Exception as err:
            raise err

    @staticmethod
    def plot_mnist_misclassified(misclassified_images, image_count=25):
      fig = plt.figure(figsize=(15, 15))
      for i in range(image_count):
        sub = fig.add_subplot(5, 5, i+1)
        plt.imshow(misclassified_images[i][0].cpu().numpy().squeeze(),cmap='gray',interpolation='none')
        sub.set_title("Predicted={0}, Actual={1}".format(str(misclassified_images[i][1].data.cpu().numpy()),str(misclassified_images[i][2].data.cpu().numpy())))
      plt.tight_layout()
      plt.show()
      fig.savefig("misclassified_images.png")
    
    @staticmethod
    def plot_cifar_misclassified(misclassified_images, image_count=25):
      classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
      fig = plt.figure(figsize=(7, 7))
      for i in range(image_count):
        sub = fig.add_subplot(5, 5, i+1)
        img =  misclassified_images[i][0].cpu()
        img = img / 2 + 0.5 
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)),cmap='gray',interpolation='nearest')
        sub.set_title("P={0}, A={1}".format(str(classes[misclassified_images[i][1].item()]),str(classes[misclassified_images[i][2].item()])))
      # plt.xlabel("A={0} ".format(str(classes[misclassified_images[i][1].item()])))
      # plt.ylabel(" P={0} ".format(str(classes[misclassified_images[i][2].item()])))
      plt.tight_layout()
      plt.show() 
      fig.savefig("cifar_misclassified_images.png")

    @staticmethod
    def image_show(img, title=None, download_image=None):
        fig = plt.figure(figsize=(7, 7))
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='none')
        if title is not None:
            plt.title(title)
        plt.pause(0.001)
        if download_image:
          fig.savefig(download_image)
          files.download(download_image)
     

    @staticmethod
    def show_cifar_classwise_image(train_loader, classes):
        iter_train_loader = iter(train_loader)
        images, labels = iter_train_loader.next()
        for i in range(len(classes)):
          index = [j for j in range(len(labels)) if labels[j] == i]
          img = torchvision.utils.make_grid(images[index[0:5]],nrow=5,padding=2,scale_each=True)
          img = img / 2 + 0.5  # unnormalize
          npimg = img.numpy()
          fig = plt.figure(figsize=(7,7))
          plt.imshow(np.transpose(npimg, (1, 2, 0)),interpolation='none')
          plt.title(classes[i])
              