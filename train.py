from torch.optim import lr_scheduler
from PIL import ImageFile
import webbrowser
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision import datasets, transforms, models
import torchvision
import torch.nn.functional as F
import cv2
# Any results you write to the current directory are saved as output.
from torchsummary import summary
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage import io, transform
import torch.optim as optim
#from logger import Logge
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')

else:
    print('CUDA is available!  Training on GPU ...')
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
train_data = datasets.ImageFolder(data_dir, transform=train_transforms)

print(len(train_data))
train_data, test_data, valid_data = torch.utils.data.random_split(train_data, [
                                                                  27298, 5850, 5850])
trainloader = torch.utils.data.DataLoader(
    train_data, batch_size=8, num_workers=1, pin_memory=True)
testloader = torch.utils.data.DataLoader(
    test_data, batch_size=8, num_workers=1, pin_memory=True)
validloader = torch.utils.data.DataLoader(
    valid_data, batch_size=8, num_workers=1, pin_memory=True)
n_classes = 2


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


model = models.densenet161()


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


total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')


total_correct = 0.0
epoches = 100
valid_loss_min = np.Inf


for epoch in range(1, epoches+1):

    torch.cuda.empty_cache()

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0

    ###################
    # train the model #
    ###################
    model.train()
    for data, target in trainloader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        scheduler.step()
        # torch.cuda.empty_cache()
        total_correct += get_num_correct(output, target)

    ######################
    # validate the model #
    ######################
    model.eval()

    for data, target in validloader:
        torch.cuda.empty_cache()
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)

        # update average validation loss
        valid_loss += loss.item()*data.size(0)
        # torch.cuda.empty_cache()

    # calculate average losses
    train_loss = train_loss/len(trainloader.sampler)
    valid_loss = valid_loss/len(validloader.sampler)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        torch.save(model.state_dict(), 'sed.pt')
        valid_loss_min = valid_loss
