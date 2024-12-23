#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 22:16:50 2024

@author: saiful
"""
# dataset link https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/code?datasetId=17810&searchQuery=torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="5,6"
import torch

import sys


if torch.cuda.is_available():
    print("CUDA (GPU support) is available in PyTorch!")
    print(f"Number of GPU(s) available: {torch.cuda.device_count()}")
    print(f"Name of the GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA (GPU support) is not available in PyTorch. Using CPU instead.")
#%% variables and transform
batchsize=100
epochs = 30 # Number of epochs
learning_rate = 0.0001

# sys.stdout = open(f"./results/console_output_lr{learning_rate}_bs{batchsize}.txt", "w")

print("learning rate  :", learning_rate )
print("batch_size :", batchsize)
# check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#%%
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets, transforms, models

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,precision_recall_fscore_support 

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm 
import pandas as pd
import os
import random
import itertools
from torch.optim.lr_scheduler import StepLR
import time
import copy
#!pip install torchinfo
# from torchinfo import summary
#!pip install seaborn
import seaborn as sns

# !pip install torch_lr_finder
# from torch_lr_finder import LRFinder

#%%
import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F 
from torch import nn,optim
from torchvision import transforms as T,datasets,models
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm
pd.options.plotting.backend = "plotly"
from torch import nn, optim
from torch.autograd import Variable

from my_models import ViTForImageClassification, ConvNextV2ForImageClassification, Swinv2ForImageClassification
from my_models import  ImageGPTForImageClassification, CvtForImageClassification
from my_models import  EfficientFormerForImageClassification, PvtV2ForImageClassification, MobileViTV2ForImageClassification
from my_models import resnet50ForImageClassification, vgg16ForImageClassification,mobilenetForImageClassification
from my_models import googlenetForImageClassification, efficientnet_b0ForImageClassification

# mode_ft = Swinv2ForImageClassification()
#%%
def data_transforms(phase = None):
    
    if phase == TRAIN:

        data_T = T.Compose([
            
                T.Resize(size = (256,256)),
                T.RandomRotation(degrees = (-20,+20)),
                T.CenterCrop(size=224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    
    elif phase == TEST or phase == VAL:

        data_T = T.Compose([

                T.Resize(size = (224,224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        
    return data_T
data_dir = "/data/saiful/pilot projects/chest_xray/chest_xray/"
TEST = 'test'
TRAIN = 'train'
VAL ='val'

trainset = datasets.ImageFolder(os.path.join(data_dir, TRAIN),transform = data_transforms(TRAIN))
testset = datasets.ImageFolder(os.path.join(data_dir, TEST),transform = data_transforms(TEST))
# validset = datasets.ImageFolder(os.path.join(data_dir, VAL),transform = data_transforms(VAL))

print('len(trainset):', len(trainset))
print('len(testset):', len(testset))


#%%
from collections import Counter

# Count the number of images for each class in the training set
train_class_counts = Counter(trainset.targets)
print("Number of images for each class in the training set:")
for class_idx, count in train_class_counts.items():
    print(f"{trainset.classes[class_idx]}: {count}")

# Count the number of images for each class in the test set
test_class_counts = Counter(testset.targets)
print("\nNumber of images for each class in the test set:")
for class_idx, count in test_class_counts.items():
    print(f"{testset.classes[class_idx]}: {count}")

#%%
class_names = trainset.classes
print(class_names)
print(trainset.class_to_idx)

#%%
trainloader = DataLoader(trainset,batch_size = 64,shuffle = True)
# validloader = DataLoader(validset,batch_size = 64,shuffle = True)
testloader = DataLoader(testset,batch_size = 64,shuffle = True)

images, labels = iter(trainloader).next()
print(images.shape)
print(labels.shape)

#%%
# for i, (images,labels) in enumerate(trainloader):
#         if torch.cuda.is_available():
#             images=Variable(images.cuda())
#             labels=Variable(labels.cuda())
            
#%%
images.shape, labels.shape



# =============================================================================
# 
# =============================================================================
def train_test_plot(method_name = "_"):
    
    print("\n\nTraining started with ", method_name)

    history=[]
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    epoch_list=[]
    
    if method_name == "Swinv2ForImageClassification":
        model_ft = Swinv2ForImageClassification().to(device)
        
    elif method_name == "ViTForImageClassification":
        model_ft = ViTForImageClassification().to(device)
        
    elif method_name == "ConvNextV2ForImageClassification":
        model_ft = ConvNextV2ForImageClassification().to(device)
        
    elif method_name == "ImageGPTForImageClassification":
        model_ft = ImageGPTForImageClassification().to(device)
 
    elif method_name == "CvtForImageClassification":
        model_ft = CvtForImageClassification().to(device)
 
    elif method_name == "EfficientFormerForImageClassification":
        model_ft = EfficientFormerForImageClassification().to(device)
        
    elif method_name == "PvtV2ForImageClassification":
        model_ft = PvtV2ForImageClassification().to(device)
            
    elif method_name == "MobileViTV2ForImageClassification":
        model_ft = MobileViTV2ForImageClassification().to(device)
        
    elif method_name == "resnet50ForImageClassification":
        model_ft = resnet50ForImageClassification().to(device)
        
    elif method_name == "vgg16ForImageClassification":
        model_ft = vgg16ForImageClassification().to(device)
                
    elif method_name == "mobilenetForImageClassification":
        model_ft = mobilenetForImageClassification().to(device)
        
    elif method_name == "googlenetForImageClassification":
        model_ft = googlenetForImageClassification().to(device)
        
    elif method_name == "efficientnet_b0ForImageClassification":
        model_ft = efficientnet_b0ForImageClassification().to(device)
        

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer  = optim.Adam(model_ft.classifier.parameters(),lr = 0.001)
    optimizer  = optim.AdamW(model_ft.parameters(),lr = learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.35)

    # Initialize variables to track the best validation accuracy
    best_valid_acc = 0.0
    best_epoch = 0
    
    for epoch in range(epochs):
        
        epoch_start = time.time()
        
        # print("Epoch: {}/{}".format(epoch+1, epochs))
        # Set to training mode
        model_ft.train()
        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
    
        # training on trainloader
        for i, (inputs, labels) in enumerate(trainloader):
            # inputs = inputs.long()
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Clean existing gradients
            optimizer.zero_grad()
            outputs = model_ft(inputs)
            ##
            if isinstance(outputs, torch.Tensor):
                # Direct tensor output
                loss = criterion(outputs, labels)
            elif hasattr(outputs, 'logits'):
                # Output has a logits attribute, typical of structured model outputs
                loss = criterion(outputs.logits, labels)
                outputs = outputs.logits
            ##
            # outputs = outputs.logits
    
            loss = criterion(outputs, labels)
            
            #
            # loss.requires_grad = True
            #
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)
            # print(" Training Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))
            
            # print("train predictions:",predictions)
            # print("train labels:", labels)
    
    
        # Validation - No gradient tracking needed
        with torch.no_grad():
            # Set to evaluation mode
            model_ft.eval()
            # Validation loop
            for j, (inputs, labels) in enumerate(testloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model_ft(inputs)
                
                ##
                if isinstance(outputs, torch.Tensor):
                    # Direct tensor output
                    loss = criterion(outputs, labels)
                elif hasattr(outputs, 'logits'):
                    # Output has a logits attribute, typical of structured model outputs
                    loss = criterion(outputs.logits, labels)
                    outputs = outputs.logits
                ##
                # outputs = outputs.logits
    
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)
                # print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
                
                
        # Find average training loss and training accuracy per epoch
        avg_train_loss = train_loss/len(trainset) 
        avg_train_acc = train_acc/float(len(trainset))
        
        # Find average validation loss and validation accuracy per epoch
        avg_valid_loss = valid_loss/len(testset) 
        avg_valid_acc = valid_acc/float(len(testset))
        
        # Update best accuracy and save model if current epoch's accuracy is higher
        if avg_valid_acc > best_valid_acc:
            best_valid_acc = avg_valid_acc
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model_ft.state_dict())
            torch.save(model_ft.state_dict(), f"/data/saiful/document_Vit/saved models/best_model_epoch_{epoch}.pt")
            print(f"Best model saved with test accuracy: {best_valid_acc*100:.4f}% at epoch {epoch+1}")

        
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        valid_loss_list.append(avg_valid_loss)
        valid_acc_list.append(avg_valid_acc)
        epoch_list.append(epoch+1)
        # history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
        epoch_end = time.time()
        # print("\n ##  Training and validation loss and  accuracy per epoch")
        print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, Validation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch+1, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
    
        # Save if the model has best accuracy till now
        # torch.save(model_ft, "/home/saiful/mobilenet_classification/saved models/document_dataset "+'_model_'+str(epoch)+'.pt')
        # torch.save(model_ft.state_dict(), "/data/saiful/document_Vit/saved models/document_dataset "+'_model_'+str(epoch)+'.pt')
        
        epoch+=1
        if epoch== epochs-1:
            print("flag1")
            print("Last epoch : ", epoch)
            break
    print("Training Finished for  ", method_name)
    
    #%%
    print("len(train_acc_list) ",len(train_acc_list))
    print("len(valid_acc_list) ",len(valid_acc_list))
    
    #%% 1
    import matplotlib.pyplot as plt
    
    # Assuming your data lists are defined: epoch_list, train_acc_list, valid_acc_list, train_loss_list, valid_loss_list
    
    # Plot for Accuracy
    train_acc_percent = [acc * 100 for acc in train_acc_list]
    valid_acc_percent = [acc * 100 for acc in valid_acc_list]
    
    # Plot for Accuracy
    plt.figure(figsize=(3.5 * 1.5, 2.8 * 1.5))
    plt.plot(epoch_list, train_acc_percent, label="Train Accuracy")
    plt.plot(epoch_list, valid_acc_percent, label="Validation Accuracy")
    plt.legend(fontsize=10)
    plt.xlabel('Epoch Number', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('Accuracy Curve', fontsize=10)
    plt.ylim(0, 100)  # Set the y-axis range from 0 to 100
    
    plt.tight_layout()
    plt.savefig(f"{method_name}_lr{learning_rate}_bs{batchsize}_document_dataset_accuracy_curve.png", dpi=300)  # Save the figure with high resolution
    plt.show()
    
    # Plot for Loss
    plt.figure(figsize=(3.5 * 1.5, 2.8 * 1.5))
    plt.plot(epoch_list, train_loss_list, label="Train Loss")
    plt.plot(epoch_list, valid_loss_list, label="Validation Loss")
    plt.legend(fontsize=10)
    plt.xlabel('Epoch Number', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('Loss Curve', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{method_name}_lr{learning_rate}_bs{batchsize}_dataset_loss_curve.png", dpi=300)
    plt.show()
    
    
    #%%
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    import numpy as np

    model_ft.load_state_dict(best_model_wts)
    model_ft.eval()  # Ensure the model is in evaluation mode
    
    true_labels = []
    predictions = []
    
    # No gradient is needed
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass to get outputs
            outputs = model_ft(inputs)
            ##
            if isinstance(outputs, torch.Tensor):
                # Direct tensor output
                loss = criterion(outputs, labels)
            elif hasattr(outputs, 'logits'):
                # Output has a logits attribute, typical of structured model outputs
                loss = criterion(outputs.logits, labels)
                outputs = outputs.logits
            ##
            # outputs = outputs.logits if hasattr(outputs, 'logits') else outputs  # Adjust based on your model's output
            
            # Convert outputs to predictions
            _, preds = torch.max(outputs, 1)
            
            # Move predictions and labels to CPU
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            
            # Collect for later evaluation
            true_labels.extend(labels)
            predictions.extend(preds)
    
    # Convert lists to numpy arrays for sklearn metrics
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    
    # Calculate metrics
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')
    accuracy = accuracy_score(true_labels, predictions)
    
    # Print metrics
    print(f'== On test data ==')
    
    print(f'Test Accuracy: {accuracy * 100:.4f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,confusion_matrix

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    # Print confusion matrix
    print('Confusion Matrix:')
    print(conf_matrix)
    
    # Optionally, plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(f"./results/{method_name}_lr{learning_rate}_bs{batchsize}_confusion_matrix.png", dpi=300)  # Save the figure with high resolution
    plt.show()


    #%%
    from sklearn.metrics import classification_report
    
    # Assuming true_labels and predictions are already defined as shown previously
    report = classification_report(true_labels, predictions, digits=4)
    
    print('classification report\n')
    print(report)
    
    #%%
    import numpy as np
    
    # Assuming true_labels and predictions are numpy arrays obtained from your test dataset
    unique_classes = np.unique(true_labels)
    class_accuracies = {}
    
    for cls in unique_classes:
        # Indices where the current class is the true label
        class_indices = np.where(true_labels == cls)
        
        # Subset of true and predicted labels where the true label is the current class
        true_for_class = true_labels[class_indices]
        preds_for_class = predictions[class_indices]
        
        # Calculate accuracy: the fraction of predictions that match the true labels for this class
        class_accuracy = np.mean(preds_for_class == true_for_class)
        
        # Store the accuracy for this class
        class_accuracies[cls] = class_accuracy
    
    # Print class-wise accuracies
    for cls, acc in class_accuracies.items():
        print(f"Class {cls}: Accuracy = {acc*100:.2f}%")
      
    print('= = = = = = = = flag 1.12 = = = = = = = = = = = = =')
    return   print('finished for ', method_name)


print('= = = = = = = = flag 1.11 = = = = = = = = = = = = =')

train_test_plot(method_name= "Swinv2ForImageClassification")
# train_test_plot(method_name= "ViTForImageClassification")
# train_test_plot(method_name= "ConvNextV2ForImageClassification")
# train_test_plot(method_name= "CvtForImageClassification")
# train_test_plot(method_name= "EfficientFormerForImageClassification")
# train_test_plot(method_name= "PvtV2ForImageClassification")
# train_test_plot(method_name= "MobileViTV2ForImageClassification")

# train_test_plot(method_name= "resnet50ForImageClassification")
# train_test_plot(method_name= "vgg16ForImageClassification")
# train_test_plot(method_name= "mobilenetForImageClassification")
# train_test_plot(method_name= "googlenetForImageClassification")
# train_test_plot(method_name= "efficientnet_b0ForImageClassification")

print('= = = = = = = = execution finished = = = = = = = = = = = = =')
