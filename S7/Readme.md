# Goal :
1. change the code such that it uses GPU
2. change the architecture to C1C2C3C40 (basically 3 MPs)
3. total RF must be more than 44
4. one of the layers must use Depthwise Separable Convolution
5. one of the layers must use Dilated Convolution
6. use GAP (compulsory):- add FC after GAP to target #of classes (optional)
7. achieve 80% accuracy, as many epochs as you want. Total Params to be less than 1M. 
8. upload to Github

## Modular Approach

To see the modular approach go [here](https://github.com/ganeshkcs/EVA4B2/tree/master/Utils).

## Loss/Accuracy Graph:

![loss_accuracy_graph](https://github.com/ganeshkcs/EVA4B2/blob/master/S7/Accuracy_Loss_Graph.png)


## 25 Misclassified Images:
 
![misclassified](https://github.com/ganeshkcs/EVA4B2/blob/master/S7/misclassified_images.png) 



### Summary:

1. Batch Size 64 and Epochs=20
2. Parameters = 549,696
3. Maximum Train accuracy: 80.39
4. Test accuracy:  83.28 
5. Receptive Field= 84







