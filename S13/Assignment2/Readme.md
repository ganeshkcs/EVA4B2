# YoloV3 - Custom training

This is to help in training custom dataset using YoloV3.  

## Dataset Preparation and Annotation

1. Download 500 images of your object
2. Clone this repo: https://github.com/miki998/YoloV3_Annotation_Tool
3. Follow the installation steps as mentioned in the repo.
4. Annotate the images using the Annotation tool.
5. After annotation, you would get label files for each of the image (in labels folder), which holds the annotation data. 

## Custom Dataset Training
![Collage of training images](https://github.com/ganeshkcs/EVA4B2/blob/master/S13/Assignment2/train_batch0.png)

1. Create a folder called weights in the root folder of this project
2. Download from: https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0
3. Place 'yolov3-spp-ultralytics.pt' file in the weights folder
4. Create a folder called 'customdata' inside 'data' folder
5. This is the final folder of 'customdata' folder :
``` data
  --customdata
    --images/
      --1.jpg
      --2.jpg
      --...
    --labels/
      --1.txt
      --2.txt
      --...
    custom.data #data file
    custom.names #your class names
    custom.txt #list of name of the images you want your network to be trained and tested on
```
 6. Copy your images to data/customdata/images folder
 7. Copy your labels (from annotation) to data/customdata/labels folder
 8. As you can see above you need to create custom.data file. For 1 class example, your file will look like this:
```
  classes=1
  train=data/customdata/custom.txt
  valid=data/customdata/custom.txt
  names=data/customdata/custom.names
```
9. Your custom_train.txt and custom_test.txt file contains path to train and test images. Your file will look like this:
```
./data/customdata/images/1.jpg
./data/customdata/images/2.jpg
./data/customdata/images/3.jpg
...
```
10. You need to add custom.names file as you can see above. For our example, we downloaded images of Jerry. Our custom.names file look like this:
```
jerry
```
11. For COCO's 80 classes, YOLOv3's output vector has 255 dimensions ( (4+1+80)*3). Now we have 1 class, so we would need to change it's architecture.
12. Copy the contents of 'yolov3-spp.cfg' file to a new file called 'yolov3-custom.cfg' file in the data/cfg folder.
13. Search for 'filters=255' (you should get entries entries). Change 255 to 18 = (4+1+1)*3
14. Search for 'classes=80' and change all three entries to 'classes=1'
15. Run this command ```python train.py --data data/customdata/custom.data --batch 20 --cache --cfg cfg/yolov3-custom.cfg --epochs 300 --nosave```
16. You can predict any images in folder by running this command ```!python detect.py --conf-thres 0.2 --source data/customdata/testImg --output out_out```

## Result after training for 300 Epochs

![result](https://github.com/santhiya-v/EVA/blob/master/S13/yolov3_training/prediction.jpg)

## Prediction of YoloV3 trained model (for object Jerry) on Video

1. https://youtu.be/qThk9ynBd_M
2. https://youtu.be/-81RADbDCtc
    
    

