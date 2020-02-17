
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision
from PIL import Image, ImageFile
from skimage import io, transform
from torch import nn
from torch import optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms




train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')

else:
    print('CUDA is available!  Training on GPU ...')
ImageFile.LOAD_TRUNCATED_IMAGES = True
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
classes = ["accident", "noaccident"]
test_loss = 0.0
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))

model.eval()
# iterate over test data
for data, target in testloader:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(
        correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(2):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(testloader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(2):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' %
              (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
