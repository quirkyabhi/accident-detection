import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision import datasets, transforms, models
import torchvision
import torch.nn.functional as F
import cv2
from torchsummary import summary
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.optim as optim
#from logger import Logge
import torch.utils.data as data_utils
train_on_gpu = torch.cuda.is_available()
import webbrowser
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
    
else:
    print('CUDA is available!  Training on GPU ...')
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.optim import lr_scheduler
#!pip install --upgrade wandb


data_dir = 'accidents'
valid_size = 0.2



train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomVerticalFlip(),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(255),
                                    #  transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       ])
valid_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomVerticalFlip(),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])




# define samplers for obtaining training and validation batches



# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir , transform=train_transforms)

print(len(train_data))
train_data,test_data, valid_data = torch.utils.data.random_split(train_data,[27298,5850,5850])
trainloader = torch.utils.data.DataLoader(train_data, batch_size=8,num_workers=1,pin_memory=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=8,num_workers=1,pin_memory=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=8,num_workers=1,pin_memory=True)
n_classes = 2




model = models.densenet161(pretrained=True)


model.classifier = nn.Sequential(nn.Linear(2208, 1000),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(1000, 2),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = model.cuda()

model.load_state_dict(torch.load('tensorboardexp.pt'))
classes=["accident","noaccident"]
#model.load_state_dict(torch.load('tensorboardexp.pt'))
count = 0
counts = 1
videopath = '12.mp4'

vid = cv2.VideoCapture(videopath)
ret = True
while ret:
    if ret==True:
        ret, frame = vid.read()

        try:
            img = Image.fromarray(frame)
        except ValueError:
            break
        except AttributeError:
            break
        img = test_transforms(img)
        img = img.unsqueeze(dim=0)
        img = img.cuda()
        model.eval()
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)



            index = int(predicted.item())
            if index==0:
                cv2.imwrite(r"D:\xamp\htdocs\img\frame%d.png" % count, frame)
                count+=1
                if counts==1:
                    webbrowser.open('127.0.0.1', new=2)
                    counts+=1
                    


            labels = 'status: ' +classes[index]
            

        cv2.putText(frame,labels,(10,100), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 5, cv2.LINE_AA)
        cv2.imshow('Frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
   
vid.release()
cv2.destroyAllWindows()    
    
    
    
  
