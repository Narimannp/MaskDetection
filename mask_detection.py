# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 22:34:42 2023

@author: narim
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
import random
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import PIL
from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.model_selection import KFold
import zipfile

from os import listdir
from os.path import isfile, join
import random
import shutil

zip_file_path = "d:/projects/mask_detection/data/gathered_data.zip"
extract_folder = "d:/projects/mask_detection/data"

# Open the ZIP file
with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    # Extract all the files to the specified folder
    zip_ref.extractall(extract_folder)    

RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
     



class_dirs = [x[0] for x in os.walk('/content/Mask_Data')][1:]


root_dir = 'd:/projects/mask_detection/data'

val_path = root_dir + '/mask_val'
test_path = root_dir + '/mask_test'
train_path = root_dir + '/mask_train'

if not os.path.isdir(root_dir):
    os.mkdir(root_dir)

if not os.path.isdir(val_path):
    os.mkdir(val_path)

if not os.path.isdir(test_path):
    os.mkdir(test_path)

if not os.path.isdir(train_path):
    os.mkdir(train_path)  
     

for cls in class_dirs:
    if not os.path.isdir(val_path + '/' + cls.split('/')[-1]):
        os.mkdir(val_path + '/' + cls.split('/')[-1])

    if not os.path.isdir(test_path + '/' + cls.split('/')[-1]):
        os.mkdir(test_path + '/' + cls.split('/')[-1])

    if not os.path.isdir(train_path + '/' + cls.split('/')[-1]):
        os.mkdir(train_path + '/' + cls.split('/')[-1]) 
os.makedirs('/content/results')
     
for clss in class_dirs:
    #print(clss)
    onlyfiles = [join(clss, f) for f in listdir(clss) if isfile(join(clss, f))]
    random.shuffle(onlyfiles)
    
    #print(onlyfiles)
    files_len = len(onlyfiles)

    train_split = 0.85
    test_split = 0.15

    train_splt = onlyfiles[:int(train_split*files_len)]
    test_splt = onlyfiles[int(train_split*files_len): ]
    #val_split = onlyfiles[int(train_split*files_len) + int(test_split*files_len) :]

    #for train_inst in train_splt:
    for img_fl in train_splt:
        final_dir = '/content/masks/mask_train'
        mask_class = img_fl.split('/')[-2]
        img_name = img_fl.split('/')[-1]

        shutil.copy(img_fl, final_dir + '/' + mask_class + '/' + img_name)

    for img_fl in test_splt:
        final_dir = '/content/masks/mask_test'
        mask_class = img_fl.split('/')[-2]
        img_name = img_fl.split('/')[-1]

        shutil.copy(img_fl, final_dir + '/' + mask_class + '/' + img_name)




train_transform = transforms.Compose(
    [transforms.Resize((124,124)),
     transforms.ColorJitter(hue=.05, saturation=.05),
     transforms.RandomHorizontalFlip(),
     transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transform = transforms.Compose(
    [transforms.Resize((124,124)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64




train_dataset = torchvision.datasets.ImageFolder(root = train_path, transform=train_transform)
test_dataset = torchvision.datasets.ImageFolder(root = test_path, transform=test_transform)

classes = ('cloth', 'n95', 'surgical', 'mask_weared_incorrect', 'without_mask')




#trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
#                                          shuffle=True, num_workers=2)


testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

k_folds = 10
kfold = KFold(n_splits=k_folds, shuffle=True)     

print("Number of data in train set = " + str(train_dataset.__len__()))
print("Number of data in test set = " + str(test_dataset.__len__()))
#print("Number of data in val set = " + str(val_dataset.__len__()))


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion*planes))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


# Function to save the model
def saveModel(epoch, fold):
    path = "/content/results/best_model_fold{}.pth".format(fold)
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
eval_losses=[]
eval_accu=[]
 
def test(epoch):
    model.eval()
 
    running_loss=0
    correct=0
    total=0
 
    with torch.no_grad():
        for data in tqdm.tqdm(testloader):
            images,labels=data[0].to(device),data[1].to(device)
       
            outputs=model(images)
 
            loss= criterion(outputs,labels)
            running_loss+=loss.item()
       
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
   
        test_loss=running_loss/len(testloader)
        accu=100.*correct/total
 
        eval_losses.append(test_loss)
        eval_accu.append(accu)
 
    return(test_loss, accu)


train_accu = []
train_losses = []
test_accu = []
test_losses = []
best_accuracy = 0.0

def train(epoch, best_accuracy):
    print('\nEpoch : %d'%epoch)
   
    model.train()
 
    running_loss=0
    correct=0
    total=0
 
    for data in tqdm.tqdm(trainloader):
     
      inputs,labels=data[0].to(device),data[1].to(device)
     
      optimizer.zero_grad()
      outputs=model(inputs)
      loss=criterion(outputs,labels)
      loss.backward()
      optimizer.step()
 
      running_loss += loss.item()


      
     
      _, predicted = outputs.max(1)
      total += labels.size(0)
      correct += predicted.eq(labels).sum().item()
       
    train_loss=running_loss/len(trainloader)
    accu=100.*correct/total
    test_loss, test_accuracy = test(epoch)
   
    train_accu.append(accu)
    train_losses.append(train_loss)
    test_accu.append(test_accuracy)
    test_losses.append(test_loss)
    print('Train Loss: %.3f | Accuracy: %.3f | Val_accuracy: %.3f | Val_loss: %.3f'%(train_loss,accu,test_accuracy,test_loss))
    if accu > best_accuracy:
        saveModel(str(epoch), str(fold))
        best_accuracy = accu

    return best_accuracy

for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset)):
    
    
    print('Fold {}'.format(fold + 1))

    train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_ids)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    valloader = DataLoader(train_dataset, batch_size=batch_size, sampler=test_sampler)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = ResNet18()
    num_class = len(classes)

    model.linear = nn.Linear(in_features = 8192, out_features = num_class)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epoch = 1

    for epoch in range(num_epoch):
        best_accuracy = train(epoch, best_accuracy)

    loss_stats = {'train': train_losses, 'val': test_losses}
    accuracy_stats = {'train': train_accu, 'val': test_accu}

    train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})

    # Plot line charts
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30,10))
    train_acc = sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
    train_loss = sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')

    fig = train_loss.get_figure()
    fig.savefig('/content/results/train_acc_fold{}.png'.format(fold), dpi=400)

    fig = train_acc.get_figure()
    fig.savefig('/content/results/train_loss_fold{}.png'.format(fold), dpi=400)

    #train_acc.savefig('/content/results/train_acc_fold{}.png'.format(fold), dpi=400)
    #train_loss.savefig('/content/results/train_loss_fold{}.png'.format(fold), dpi=400)

    y_pred_list = []
    y_true_list = []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in tqdm.tqdm(testloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_test_pred = model(x_batch)
            _, y_pred_tag = torch.max(y_test_pred, dim = 1)
            y_pred_list.append(y_pred_tag.cpu().numpy())
            y_true_list.append(y_batch.cpu().numpy())

    y_pred_final = []
    y_true_final = []
    for i in y_pred_list:
        y_pred_final = y_pred_final + list(i)

    for i in y_true_list:
        y_true_final = y_true_final + list(i)

    print(classification_report(y_true_final, y_pred_final))
    
    plt.figure(figsize = (10,10))
    cm_mat = sns.heatmap(confusion_matrix(y_true_final, y_pred_final), annot=True)
    fig = cm_mat.get_figure()
    fig.savefig('/content/results/cm_fold{}.png'.format(fold), dpi=400)