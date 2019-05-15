from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

def process_image(image):
    """ 
        Scales, crops, and normalizes a PIL image for a PyTorch model.
        Returns:
            Numpy array
    """
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(means,stds)
    ])

    return trans(image).numpy()

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
