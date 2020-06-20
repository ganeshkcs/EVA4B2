# Goal :

1. Write a code that draws one cycle triangle.

2. Writting Code in below architecture for Cifar10:
 PrepLayer:
    Conv 3x3 s1, p1) >> BN >> RELU [64k]
 Layer1:
     X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
     R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
     Add(X, R1)
     Layer 2:
     Conv 3x3 [256k]
     MaxPooling2D
     BN
     ReLU
 Layer 3:
     X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
     R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
     Add(X, R2)
     MaxPooling with Kernel Size 4
     FC Layer 
     SoftMax
 3. Use One Cycle Policy such that:
     Total Epochs = 24
     Max at Epoch = 5
     LRMIN = FIND
     LRMAX = FIND
     NO Annihilation
 4. Use transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
 5. Use Batch size = 512
 6. Achieve Target Accuracy: 90%. 
 
 ### Albumentaion applied images 
![albumentaion1](https://github.com/ganeshkcs/EVA4B2/blob/master/S11/images/bird_cutout.png)

### Finding Max LR using range test with inputs max lr:1e-1, start lr:1e-4 and num of iterations:980 i.e 10 epochs
![lr_finder](https://github.com/ganeshkcs/EVA4B2/blob/master/S11/images/acc_lr.png

### One Cycle triangle using all trained lr for 24 epochs
![one_cycle](https://github.com/ganeshkcs/EVA4B2/blob/master/S11/images/one_cycle_training.png)

### Train and Validation Accuracy Change:
![accuracy_graph](https://github.com/ganeshkcs/EVA4B2/blob/master/S11/images/train_val.png)

### Accuracy and Loss for both Train and Validation:
![accuracy_graph](https://github.com/ganeshkcs/EVA4B2/blob/master/S11/images/all_val_acc.png)

### GradCam and heatmap visualization for 25 missclassified images:
![misclassified1](https://github.com/ganeshkcs/EVA4B2/blob/master/S11/images/gradcam_misclassified.png)

### Summary:
1. Used L2 Regularization, scheduler OneCycleLR with max LR value `0.044422151755141255` which
 is found using lr finder range test.
2. Batch Size 512 and Epochs=24
3. Parameters = 6,573,120
4. Applied albumentations for train set such as Padding, Random Crop, Normalize, HorizontalFlip, Cutout and ToTensor.
5. Applied albumentations for test set such as Normalize and ToTensor.
7. Maximum Train accuracy:  `97.63`
8. Test accuracy:  `93.18`

