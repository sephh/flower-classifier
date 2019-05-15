import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import torch.nn.functional as F

import numpy as np

def load_data(data_dir = 'flowers'):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, data_transforms['test'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=True)
    }
    
    return dataloaders, image_datasets, data_transforms

def train(model, dataloaders, device, lr=0.001, epochs=5):
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    criterion = nn.NLLLoss()
    train_len = len(dataloaders["train"]);
    
    model.to(device)
    model.train()
    
    print(f'Training start with {epochs} epochs')

    for e in range(epochs):
        running_loss = 0
        progress = 0

        for images, labels in dataloaders['train']:
            images, labels = images.to(device), labels.to(device);

            optimizer.zero_grad()

            logits = model.forward(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress += 1
            if progress % 10 == 0:
                print(f'Training epoch {e+1} progress: {(progress/train_len)*100:.2f}%')

        else:
            valid_loss = 0
            accuracy = 0

            with torch.no_grad():
                model.eval()

                for images, labels in dataloaders['valid']:
                    images, labels = images.to(device), labels.to(device)

                    logits = model(images)
                    loss = criterion(logits, labels)

                    ps = torch.exp(logits)

                    top_k, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
                    valid_loss += loss.item()

            model.train()

            print(
                f'Epoch {e+1} / {epochs}',
                f'Training Loss: {running_loss/train_len:.3f}',
                f'Validation Loss: {valid_loss/len(dataloaders["valid"]):.3f}',
                f'Accuracy: {accuracy/len(dataloaders["valid"]):.3f}',
            )
    
    print('Training end')
    
def run_test(model):
    criterion = nn.NLLLoss()
    valid_loss = 0
    accuracy = 0
    
    with torch.no_grad():
                model.eval()

                for images, labels in dataloaders['test']:
                    images, labels = images.to(device), labels.to(device)

                    logits = model(images)
                    loss = criterion(logits, labels)

                    ps = torch.exp(logits)

                    top_k, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
                    valid_loss += loss
                
                print(
                    f'Test Loss: {valid_loss/len(dataloaders["test"]):.3f}',
                    f'Test Accuracy: {accuracy/len(dataloaders["test"]):.3f}',
                )
                
def get_train_args():
    """
    Retrieves and parses command line arguments provided by the user when
    they run the program from a terminal window.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('data_dir', type=str, help='The directory with flower images', default='flowers')
    parser.add_argument('--save_dir', type=str, help='The directory where checkpoint is saved', default='checkpoint')
    parser.add_argument('--arch', type=str, help='The CNN model architecture. Available options are resnet, alexnet and vgg ', default='vgg')
    parser.add_argument('--learning_rate', type=float, help='The training lr', default=0.001)
    parser.add_argument('--dropout', type=float, help='classifier dropout', default=0.2)
    parser.add_argument('--hidden_units', type=int, help='Hidden units', default=512)
    parser.add_argument('--epochs', type=int, help='Hidden units', default=5)
    parser.add_argument('--gpu', action="store_true", default=True)
    
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    args = parser.parse_args()
    
    return args