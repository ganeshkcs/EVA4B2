import torch.nn as nn
import torch.nn.functional as F
class newModel(nn.Module):
    def __init__(self):
        super(newModel, self).__init__()

        self.prepLayer = nn.Sequential(
          nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), 
          nn.BatchNorm2d(64),  
          nn.ReLU(),
          )
        self.layer1 = nn.Sequential(
          nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
          nn.MaxPool2d(2, 2),
          nn.BatchNorm2d(128),
          nn.ReLU(),  
          )

        self.R1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), 
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
               
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            )

        self.R2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False), 
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
               
            )
        

        self.pool = nn.MaxPool2d(4, 4)
        self.fc =  nn.Linear(in_features = 512, out_features = 10, bias=False)

    def forward(self, x):

        prepLayer = self.prepLayer(x)
        x1 = self.layer1(prepLayer)
        r1 = self.R1(x1)
        layer1 = x1 + r1
        layer2 = self.layer2(layer1)
        x2 = self.layer3(layer2)
        r2 = self.R2(x2)
        layer3 = x2 + r2
        x4 = self.pool(layer3) 
        x5 = x4.view(x4.size(0), -1)
        x = self.fc(x5) 
        return x #not using softmax as we are using crossentopy loss