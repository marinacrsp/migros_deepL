import os
import torch
from tqdm import tqdm
from torcheval.metrics.aggregation.auc import AUC 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn
from sklearn import metrics
import numpy as np
import torchvision.models as models

compute_auc = AUC()
softmax = nn.Softmax(dim=1)

def build_model(pretrained=True, fine_tune=True, num_classes=2):
    if pretrained:
        print("[INFO]: Loading pre-trained weights")
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif not pretrained:
        print("[INFO]: Not loading pre-trained weights")
        model = models.resnet18(weights=None)
    if fine_tune:
        # Freeze all layers first
        for param in model.parameters():
            param.requires_grad_(False)
        print("[INFO]: Fine-tuning the following layers...")
        for module, _ in model.named_children():
            layer_block = False
            if 'layer' in module:
                if 'layer1' in module:
                    layer_block = model.layer1
                elif 'layer2' in module:
                    layer_block = model.layer2
                # if 'layer3' in module:
                #     layer_block = model.layer3
                # elif 'layer4' in module:
                #     layer_block = model.layer4
                
                if layer_block:
                    for buffer in layer_block:
                        buffer.conv1.requires_grad_(True)
                        print(buffer.conv2)
        # for params in model.parameters():
        #     params.requires_grad = True
    elif not fine_tune:
        print("[INFO]: Freezing hidden layers...")
        for params in model.parameters():
            params.requires_grad = False

    # Change the final classification head, it is trainable.
    model.fc = nn.Linear(512, num_classes)
    return model

# Training function.
def train(model, trainloader, optimizer, criterion, device):
    model.train()
    print("Training")
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        
        image = data['instance_images']
        labels = data['instance_labels']
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        
        train_running_correct += (preds == labels).sum().item()

        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100.0 * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

def test(model, testloader, criterion, device):
    model.eval()
    print("Testing")
    test_running_loss = 0.0
    test_running_correct = 0   
    counter = 0
    predictions = []
    gt = []
    auc_scores = []
    for i, data in tqdm(enumerate(testloader), total=len(testloader)):
        counter += 1
        image = data['instance_images']
        labels = data['instance_labels']
        image = image.to(device)
        labels = labels.to(device)
        
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        test_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)

        y_probabilities = softmax(outputs.data)
        #print(y_probabilities)
        y_preds = y_probabilities[:,1]
        #print(y_preds)
        
        fpr, tpr, _ = metrics.roc_curve(labels.cpu(), y_preds.cpu())
        auc_scores.append(metrics.auc(fpr, tpr))

        test_running_correct += (preds == labels).sum().item() # TP + TN
        predictions.append(preds)
        gt.append(labels)

    # Loss and accuracy for the complete epoch.
    final_loss = test_running_loss / counter
    final_accuracy = 100.0 * (test_running_correct / len(testloader.dataset))
    return final_loss, final_accuracy, predictions, gt, np.mean(auc_scores)

# Validation function.
def validate(model, testloader, criterion, device):
    model.eval()
    print("Validation")
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    predictions = []
    gt = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            image = data['instance_images']
            labels = data['instance_labels']
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
            predictions.append(preds)
            gt.append(labels)
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100.0 * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc, predictions, gt