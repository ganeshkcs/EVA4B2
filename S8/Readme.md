
# Goal :
1. Go through this repository: https://github.com/kuangliu/pytorch-cifar
2. Extract the ResNet18 model from this repository and add it to your API/repo. 
3. Use data loader, model loading, train, and test code to train ResNet18 on Cifar10
4. Achieve Target is 85% accuracy. No limit on the number of epochs. Use default ResNet18 code (so params are fixed). 
5. one of the layers must use Dilated Convolution


## Accuracy/Loss:

![accuracy_loss_graph](https://github.com/ganeshkcs/EVA4B2/blob/master/S8/Loss-Accuracy.png)



## 25 Misclassified Images:
 
![misclassified](https://github.com/ganeshkcs/EVA4B2/blob/master/S8/misclassified.png) 

### Summary:
1. Used L2 Regularization, schedular OneCycleLR
2. Batch Size 128 and Epochs=20
3. Parameters = 11,173,962
3. Maximum Train accuracy:  94.23
4. Test accuracy:  90.33




