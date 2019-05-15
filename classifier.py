import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F

# Load available models
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

available_models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}

class Classifier(nn.Module):
    
    def __init__(self, hidden_layer_size, dropout):
        super().__init__();
        
        self.fc1 = nn.Linear(25088, 1024)
        self.fc2 = nn.Linear(1024, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, 102)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        
        x = self.dropout(
            F.relu(self.fc1(x))
        )
        x = self.dropout(
             F.relu(self.fc2(x))
        )
        x = self.dropout(
             F.log_softmax(self.fc3(x), dim=1)
        )
        
        return x
    
def get_pretrained_model(arch, hidden_units=512, dropout=0.2):
    model = available_models.get(arch, available_models['vgg'])

    for param in model.parameters():
        param.requires_grad = False

    classifier = Classifier(hidden_units,dropout)
    classifier.eval()
    
    model.classifier = classifier
    
    return model, hidden_units, dropout

def save_model(model, arch, hidden_units, dropout, class_to_idx, epochs=5, path='checkpoint'):
    model.to('cpu')
    
    checkpoint = {
        'class_to_idx': class_to_idx,
        'state_dict': model.state_dict(),
        'epochs': epochs,
        'hidden_units': hidden_units, 
        'dropout': dropout,
        'arch': arch,
    }
    
    torch.save(checkpoint, path + '/checkpoint.pth')

def load_checkpoint(checkpoint_path='checkpoint/checkpoint.pth'):
    checkpoint = torch.load(checkpoint_path)
    model = available_models.get(checkpoint['arch'], available_models['vgg'])
    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = Classifier(checkpoint['hidden_units'], checkpoint['dropout'])
    classifier.eval()
    
    model.classifier = classifier
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model