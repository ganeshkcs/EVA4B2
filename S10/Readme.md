# Goal :
1. Apply CutOut
2. Implement LR Finder for SGD with Momentum.
3. Implement ReduceLROnPlatea
4. Train for 50 epochs.
5. Run GradCAM on the any 25 misclassified images. 

## Albumentaion applied images for all cifa10 target classes
![albumentaion1](https://github.com/ganeshkcs/EVA4B2/blob/master/S10/images/bird_cutout.png)
![albumentaion2](https://github.com/ganeshkcs/EVA4B2/blob/master/S10/images/car_cutout.png)
![albumentaion3](https://github.com/ganeshkcs/EVA4B2/blob/master/S10/images/deer_cutout.png)
![albumentaion4](https://github.com/ganeshkcs/EVA4B2/blob/master/S10/images/dog_cutout.png)
![albumentaion5](https://github.com/ganeshkcs/EVA4B2/blob/master/S10/images/frog_cutout.png)
![albumentaion6](https://github.com/ganeshkcs/EVA4B2/blob/master/S10/images/horse_cutout.png)
![albumentaion7](https://github.com/ganeshkcs/EVA4B2/blob/master/S10/images/plane_cutout.png)
![albumentaion8](https://github.com/ganeshkcs/EVA4B2/blob/master/S10/images/ship_cutout.png)
![albumentaion9](https://github.com/ganeshkcs/EVA4B2/blob/master/S10/images/truck_cutout.png)
![albumentaion10](https://github.com/ganeshkcs/EVA4B2/blob/master/S10/images/cat_cutout.png)

## GradCam and Heatmap visualization for 25 misclassified images in all layers
![misclassified1](https://github.com/ganeshkcs/EVA4B2/blob/master/S10/images/misclassified_gradcam.png)

## Train and Validation Accuracy Change:
![accuracy_graph](https://github.com/ganeshkcs/EVA4B2/blob/master/S10/images/accuracy.png)

## Accuracy and Loss for both Train and Validation:
![accuracy_graph](https://github.com/ganeshkcs/EVA4B2/blob/master/S10/images/accuracy_loss.png)

### Summary:
1. Used L2 Regularization, scheduler ReduceLROnPlatea.
2. Batch Size 128 and Epochs=50
3. Parameters = 11,173,962
4. Applied albumentations for train set such as Normalize, HorizontalFlip, Cutout and ToTensor
5. Applied albumentations for test set such as Normalize and ToTensor
6. Implemented GradCam function as a module. 
7. Maximum Train accuracy:   91.76
8. Maximum Test accuracy:  90.33
