# Goal :
1. Apply Albumentaion
2. Implement GradCam function as a module. 

## Albumentaion applied images for cifa10 classes
![albumentaion1](https://github.com/ganeshkcs/EVA4B2/blob/master/S9/images/bird_cutout.png)
![albumentaion2](https://github.com/ganeshkcs/EVA4B2/blob/master/S9/images/car_cutout.png)
![albumentaion3](https://github.com/ganeshkcs/EVA4B2/blob/master/S9/images/cat_cutout.png)
![albumentaion4](https://github.com/ganeshkcs/EVA4B2/blob/master/S9/images/dog_cutout.png)
![albumentaion5](https://github.com/ganeshkcs/EVA4B2/blob/master/S9/images/plane_cutout.png)


## GradCam and heatmap visualization of an image in layer1, layer2, layer3, layer4
![layer1](https://github.com/ganeshkcs/EVA4B2/blob/master/S9/images/truck_alllayers_layer1.png)
![layer2](https://github.com/ganeshkcs/EVA4B2/blob/master/S9/images/truck_alllayers_layer2.png)
![layer3](https://github.com/ganeshkcs/EVA4B2/blob/master/S9/images/truck_alllayers_layer3.png)
![layer4](https://github.com/ganeshkcs/EVA4B2/blob/master/S9/images/truck_alllayers_layer4.png)

## GradCam and heatmap visualization for few correctly classified images in layer4
![correctclassified1](https://github.com/ganeshkcs/EVA4B2/blob/master/S9/images/airplane_layer4.png)
![correctclassified2](https://github.com/ganeshkcs/EVA4B2/blob/master/S9/images/bird_layer4.png)
![correctclassified3](https://github.com/ganeshkcs/EVA4B2/blob/master/S9/images/truck_layer4.png)

## GradCam and heatmap visualization for few misclassified classified images in layer4
![misclassified1](https://github.com/ganeshkcs/EVA4B2/blob/master/S9/images/bird_misclassified.png)
![misclassified2](https://github.com/ganeshkcs/EVA4B2/blob/master/S9/images/dog_misclassified.png)
![misclassified2](https://github.com/ganeshkcs/EVA4B2/blob/master/S9/images/ship_misclassified.png)
![misclassified2](https://github.com/ganeshkcs/EVA4B2/blob/master/S9/images/truck_misclassified.png)


## Loss/Accuracy:

![accuracy_graph](https://github.com/ganeshkcs/EVA4B2/blob/master/S9/images/accuracy_loss.png)


## 25 Misclassified Images:
 
![misclassified](https://github.com/ganeshkcs/EVA4B2/blob/master/S9/images/misclassified.png) 




### Summary:
1. Used L2 Regularization, schedular OneCycleLR
2. Batch Size 128 and Epochs=20
3. Parameters = 11,173,962
4. Applied albumentations for train set such as Normalize, HorizontalFlip, Cutout and ToTensor
5. Applied albumentations for test set such as Normalize and ToTensor
6. Implement GradCam function as a module. 
7. Maximum Train accuracy:   90.03
8. Test accuracy:  90.58





