In this code the design of the convolution network is approached to achieve the target validation accuracy of 99.4% with less than 20k parameters and 20 epochs. The below mentioned are the steps taken to reach the target.

Network with less than 20k parameters achieved with having the number of kernels (3*3) as 8,16,32 and (1*1) as 8,10.
Adding Max Pooling.
Adding GAP layer.
Adding Batch Normalization and Dropout. 
Keeping the batch size to 128. 

Details :

Total no. of parameters : 12,274 
Validation accuracy : 99.48% (14th epoch)
