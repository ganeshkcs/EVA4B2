# 25 Misclassified Images for below cases

## Without L1/L2 with BN

![](https://github.com/ganeshkcs/EVA4B2/blob/master/S6/BN_MISCLASSIFIED_WITHOUT_L1L2.png)

## Without L1/L2 with GBN

![](https://github.com/ganeshkcs/EVA4B2/blob/master/S6/GBN_MISCLASSIFIED_WITHOUT_L1L2.png)

# Validation accuracy and losses are calculated for the below cases
1. without L1/L2 with BN
2. without L1/L2 with GBN
3. with L1 with BN
4. with L1 with GBN
5. with L2 with BN
6. with L2 with GBN
7. with L1 and L2 with BN
8. with L1 and L2 with GBN

## Validation Accuracy Graph of all above 8 models.

![](https://github.com/ganeshkcs/EVA4B2/blob/master/S6/LOSS_ACCURACY_FOR_8CASES.png)

### Analysis:

Both in GBN and BN the accuracy is best when the models are run "Without L1L2". The max accuracy achieved in both the cases are 99.45(GBN 22nd EPOCH without L1L22) and 99.54(BN 14th Epoch without L1L2).

L1,L2 and L1L2 seems to be spiking up and down for both GBN and BN.

Applying L1 ,L2 and L1L2 Regularisation seems to decrease the accuracy(in my case).
