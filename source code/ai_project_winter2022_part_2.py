# -*- coding: utf-8 -*-
"""AI_Project_Winter2022.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1X5tPwOz743a6xWXyi5ce8VrR8XhxWU8B
"""

import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, random_split, DataLoader, SubsetRandomSampler, ConcatDataset
from torch import nn
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay

from google.colab import drive
drive.mount('/content/drive')

# Some predefine variable for different path

# Modify below variable according to your need
dataset_path = "drive/MyDrive/COMP6721_project/AI" # This contains path to the actual dataset
model_path = "./model.pth" # This contains path to the actual model
test_image_path = r"drive/MyDrive/COMP6721_project/test_image" # This contains path for images to test, don't remove 'r'
# dataset_path = "E:\Concordia\Winter2022\COMP6721-AI\Assignments\Project\project part 1\project\dataset" # This contains path to the actual dataset
# model_path = "E:\Concordia\Winter2022\COMP6721-AI\Assignments\Project\project part 1\project\model\model.pth" # This contains path to the actual model
# test_image_path = r"E:\Concordia\Winter2022\COMP6721-AI\Assignments\Project\project part 1\project\test_image" # This contains path for images to test, don't remove 'r'

dataset = ImageFolder(dataset_path)
print(f'number of images: {len(dataset)}')
print(f'number of classes: {len(dataset.classes)}')
target_names = dataset.classes
print(target_names)

# Here we are splitting our dataset into 1500 training and 500 testing images
test_pct = 0.25
test_size = int(len(dataset)*test_pct)
train_size = len(dataset) - test_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])
valid_pct = 0.10
valid_size = int(len(train_ds)*valid_pct)
train_size = train_size - valid_size
train_ds, valid_ds = random_split(train_ds,[train_size,valid_size])

class FaceMaskDataset(Dataset):
    
    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        img, label = self.ds[idx]
        if self.transform:
            img = self.transform(img)  
            return img, label

batch_size = 15

train_transform = transforms.Compose([
        transforms.Resize((96,96)),
        transforms.ToTensor(),
    ])

test_transform = transforms.Compose([
        transforms.Resize((96,96)),
        transforms.ToTensor()
    ])

train_dataset = FaceMaskDataset(train_ds, train_transform)
valid_dataset = FaceMaskDataset(valid_ds, train_transform)
test_dataset = FaceMaskDataset(test_ds, test_transform)

train_dl = DataLoader(train_dataset, batch_size, shuffle=True)

valid_dl = DataLoader(valid_dataset, batch_size)
test_dl = DataLoader(test_dataset, batch_size)

# This method is use to train model on 1500 images we have splitted
def Train(model,optimizer,dataloader,device):
    loss_tracker = []
    Train_accuracy_tracker = []
    correct = total = 0
    for i,(data,label) in enumerate(dataloader):
        data = data.requires_grad_()
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        
        outputs= model(data)
        CE_loss = nn.CrossEntropyLoss()
        loss = CE_loss(outputs,label)
        loss.backward()  

        optimizer.step()

        
        with torch.no_grad():
                _,predicted = torch.max(outputs.data,1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                accuracy = (correct/total)*100
        loss_tracker.append(loss.item())
        Train_accuracy_tracker.append(accuracy)

    return loss_tracker, Train_accuracy_tracker

# Here we will test our input images and 500 testing images
def Test(model,dataloader,device):
    loss_tracker = []
    Test_accuracy_tracker = []
    total = correct = 0
    predict = []
    labels = []
    for i,(data,label) in enumerate(dataloader):
        data = data.to(device)
        label.to(device)
        labels.extend(label)
        with torch.no_grad():
         
            output = model(data)
            CE_loss = nn.CrossEntropyLoss()
            loss = CE_loss(output,label)
            _,predicted = torch.max(output.data,1)
            predict.extend(predicted)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            accuracy = (correct/total)*100
            
        loss_tracker.append(loss.item())
        Test_accuracy_tracker.append(accuracy)
        
    return sum(loss_tracker)/len(loss_tracker), sum(Test_accuracy_tracker)/len(Test_accuracy_tracker), labels, predict

class conv_net(nn.Module):
    def __init__(self):
        super(conv_net,self).__init__()
        self.network = nn.Sequential(
                                      nn.Conv2d(3, 16, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3),
                                     
                                      nn.Conv2d(16, 32, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2),
                                     
                                      nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3),
                                     
                                      nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2),
                                      
                                      nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2),
                                      
                                      nn.Flatten(),
                                      nn.Linear(256, 1024),
                                      nn.Linear(1024, 64),
                                      nn.Linear(64, 5),
                                      nn.Softmax(1)


                                    )
        
    
    def forward(self,x):
        x = x.to(device)
        return self.network(x)

device = "cpu"

splits = KFold(n_splits=10, shuffle=True)

dataset_d = ConcatDataset([train_dataset, test_dataset])

overall_loss = []
overall_acuracy = []

for fold, (train_idx,val_idx) in enumerate(splits.split(torch.arange(len(dataset_d)))):
  epochs = 20
  learning_rate = 0.0005

  print(f'device: {device}')

  model = conv_net()
  model.to(device)

  optimizer = torch.optim.RAdam(model.parameters(), learning_rate)

  train_loss_tracker = []
  train_accuracy_tracker = []

  test_loss_tracker = []
  test_accuracy_tracker = []
  
  print('Fold {}'.format(fold+ 1))
  train_sampler= SubsetRandomSampler(train_idx)

  test_sampler= SubsetRandomSampler(val_idx)

  train_loader= DataLoader(dataset_d, batch_size=batch_size, sampler=train_sampler)
  valid_accuracy = 0.0
  valid_loader= DataLoader(dataset_d, batch_size=batch_size, sampler=test_sampler)
  
  for epoch in range(epochs):
    print(f'epoch: {epoch}')
    train_loss, train_accuracy = Train(model,optimizer,train_loader,device)
    valid_loss, valid_accuracy, labels, predict = Test(model,valid_loader,device)
    train_loss_tracker.extend(train_loss)
    train_accuracy_tracker.extend(train_accuracy)
    test_loss_tracker.append(valid_loss)
    test_accuracy_tracker.append(valid_accuracy)
    
    overall_loss.append(valid_loss)
    overall_acuracy.append(valid_accuracy)

    print('\t training loss/accuracy: {0:.2f}/{1:.2f}'.format(sum(train_loss)/len(train_loss), sum(train_accuracy)/len(train_accuracy)))
    print('\t validation loss/accuracy: {0:.2f}/{1:.2f}'.format(valid_loss, valid_accuracy))
  
  print(classification_report(labels, predict, target_names=target_names))
  cm = confusion_matrix(labels,predict)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names)
  disp.plot(xticks_rotation="vertical")
  plt.show()
  
  model_name = "model/" + str(valid_accuracy) + ".pth"
  torch.save(model, model_name)

gen_model = torch.load(model_path)
valid_loss, valid_accuracy, labels, predict = Test(gen_model,test_dl,device)
# Classification report and Confusion matrix for testing images
print(classification_report(labels, predict, target_names=target_names))
cm = confusion_matrix(labels,predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names)
disp.plot(xticks_rotation="vertical")
plt.show()

# Testing part for pre trained model
gen_model = torch.load(model_path)
# test_image_path = test_image_path.replace("\\","/")
test_image_path = r"drive/MyDrive/COMP6721_project/test_image/women"
image = ImageFolder(test_image_path)
test_image = FaceMaskDataset(image, test_transform)
test_data = DataLoader(test_image, 10)
test_loss , test_accuracy,l,p = Test(gen_model,test_data,device)

# Classification report and Confusion matrix for testing images
print(classification_report(l,p, target_names=target_names))
cm = confusion_matrix(l,p)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names)
disp.plot(xticks_rotation="vertical")
plt.show()

gen_model = torch.load(model_path)

test_image_path = r"drive/MyDrive/COMP6721_project/test_image/men"
image = ImageFolder(test_image_path)
test_image = FaceMaskDataset(image, test_transform)
test_data = DataLoader(test_image, 10)
test_loss , test_accuracy,l,p = Test(gen_model,test_data,device)

# Classification report and Confusion matrix for testing images
print(classification_report(l,p, target_names=target_names))
cm = confusion_matrix(l,p)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names)
disp.plot(xticks_rotation="vertical")
plt.show()

gen_model = torch.load(model_path)

test_image_path = r"drive/MyDrive/COMP6721_project/test_image/Asian_race"
image = ImageFolder(test_image_path)
test_image = FaceMaskDataset(image, test_transform)
test_data = DataLoader(test_image, 10)
test_loss , test_accuracy,l,p = Test(gen_model,test_data,device)

# Classification report and Confusion matrix for testing images
print(classification_report(l,p, target_names=target_names))
cm = confusion_matrix(l,p)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names)
disp.plot(xticks_rotation="vertical")
plt.show()

gen_model = torch.load(model_path)
test_image_path = r"drive/MyDrive/COMP6721_project/test_image/African_race"
image = ImageFolder(test_image_path)
test_image = FaceMaskDataset(image, test_transform)
test_data = DataLoader(test_image, 10)
test_loss , test_accuracy,l,p = Test(gen_model,test_data,device)

# Classification report and Confusion matrix for testing images
print(classification_report(l,p, target_names=target_names))
cm = confusion_matrix(l,p)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names)
disp.plot(xticks_rotation="vertical")
plt.show()

gen_model = torch.load(model_path)
test_image_path = r"drive/MyDrive/COMP6721_project/test_image/Whitish_race"
image = ImageFolder(test_image_path)
test_image = FaceMaskDataset(image, test_transform)
test_data = DataLoader(test_image, 10)
test_loss , test_accuracy,l,p = Test(gen_model,test_data,device)

# Classification report and Confusion matrix for testing images
print(classification_report(l,p, target_names=target_names))
cm = confusion_matrix(l,p)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names)
disp.plot(xticks_rotation="vertical")
plt.show()