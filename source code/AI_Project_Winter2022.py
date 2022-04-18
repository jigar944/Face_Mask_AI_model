#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, random_split, DataLoader
from torch import nn
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[20]:


# Some predefine variable for different path

# Modify below variable according to your need
dataset_path = "D:\Study\COMP6721\Project\project_final\dataset" # This contains path to the actual dataset
model_path = "D:\Study\COMP6721\Project\project_final\model\model.pth" # This contains path to the actual model
test_image_path = r"D:\Study\COMP6721\Project\project_final\test_image" # This contains path for images to test, don't remove 'r'


# In[3]:


dataset = ImageFolder(dataset_path)
print(f'number of images: {len(dataset)}')
print(f'number of classes: {len(dataset.classes)}')
target_names = dataset.classes
print(target_names)


# In[4]:


# Here we are splitting our dataset into 1500 training and 500 testing images
test_pct = 0.25
test_size = int(len(dataset)*test_pct)
train_size = len(dataset) - test_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])
print(len(train_ds))


# In[5]:


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


# In[6]:


batch_size = 100

train_transform = transforms.Compose([
   transforms.Resize((50,50)),
    transforms.ToTensor()    
])


test_transform = transforms.Compose([
    transforms.Resize((50,50)), 
    transforms.ToTensor()
])

train_dataset = FaceMaskDataset(train_ds, train_transform)
test_dataset = FaceMaskDataset(test_ds, test_transform)

train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size)


# In[7]:


# This method is use to train model on 1500 images we have splitted
def Train(model,optimizer,dataloader,device):
    loss_tracker = []
    Train_accuracy_tracker = []
    correct = total = 0
    for i,(data,label) in enumerate(dataloader):
        data = data.requires_grad_()
        data.to(device)
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
        data.to(device)
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
        
    return sum(loss_tracker)/len(loss_tracker), sum(Test_accuracy_tracker)/len(Test_accuracy_tracker),labels,predict
        


# In[8]:


class conv_net(nn.Module):
    def __init__(self):
        super(conv_net,self).__init__()
        self.network = nn.Sequential(
                                      nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1,bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                     
                                      nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1,bias=True), 
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                     
                                      nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1,bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                     
                                      nn.Dropout(p=0.3),
                                      nn.Flatten(),
                                      nn.Linear(48 * 5 * 5, 196),
                                      nn.Linear(196, 5)
                                    )
        
    
    def forward(self,x):
        return self.network(x)


# In[9]:


epochs = 50
learning_rate = 0.001

device = "cpu"
print(f'device: {device}')


model = conv_net()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), learning_rate)

train_loss_tracker = []
train_accuracy_tracker = []

test_loss_tracker = []
test_accuracy_tracker = []

for epoch in range(epochs):
    print(f'epoch: {epoch}')
    train_loss,train_accuracy = Train(model,optimizer,train_dl,device)
    test_loss , test_accuracy,labels,predict = Test(model,test_dl,device)
    train_loss_tracker.extend(train_loss)
    train_accuracy_tracker.extend(train_accuracy)
    test_loss_tracker.append(test_loss)
    test_accuracy_tracker.append(test_accuracy)
    
    print('\t training loss/accuracy: {0:.2f}/{1:.2f}'.format(sum(train_loss)/len(train_loss), sum(train_accuracy)/len(train_accuracy)))
    print('\t testing loss/accuracy: {0:.2f}/{1:.2f}'.format(test_loss, test_accuracy))

torch.save(model, model_path)


# In[10]:


# Classification report and Confusion matrix for our dataset and trained model
print(classification_report(labels,predict, target_names=target_names))
print(confusion_matrix(labels,predict))


# In[23]:


# Testing part for pre trained model
gen_model = torch.load(model_path) 
test_image_path = test_image_path.replace("\\","/")
image = ImageFolder(test_image_path)
test_image = FaceMaskDataset(image, test_transform)
test_data = DataLoader(test_image, 10)
test_loss , test_accuracy,l,p = Test(gen_model,test_data,device)


# In[24]:


# Classification report and Confusion matrix for testing images
print(classification_report(l,p, target_names=target_names))
print(confusion_matrix(l,p))

