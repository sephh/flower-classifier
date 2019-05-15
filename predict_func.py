import argparse

import torch
import utils
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from classifier import load_checkpoint

def predict(image_path, device, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''    
    pil_image = Image.open(image_path)
    np_image = utils.process_image(pil_image)
    image_tensor = torch.tensor(np_image).unsqueeze(0).to(device)
    image_tensor = image_tensor.float()
    
    with torch.no_grad():
        logits = model.forward(image_tensor)
        ps = torch.exp(logits.data)
    
    return ps.topk(topk)

def make_a_prediction(checkpoint, device, topk, cat_to_name, image_path= './flowers/test/1/image_06743.jpg'):
    model = load_checkpoint(checkpoint)
    model.to(device)

    with open(cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
    
    probs, classes = predict(
        image_path=image_path, 
        device=device, 
        model=model, 
        topk=topk
    )

    objects = classes.cpu().data.numpy().squeeze()
    category_names = [cat_to_name[str(i+1)] for i in objects]
    performance = probs.cpu().data.numpy().squeeze()
    
#     y_pos = np.arange(topk)
#     plt.barh(y_pos, performance, align='center', alpha=0.5)
#     plt.yticks(y_pos, [cat_to_name[str(i+1)] for i in objects])

#     plt.show()
    print('PREDICTION:', '\n')
    
    for label, prob in zip(category_names, performance):
        print(f'Chance to be a {label} is {prob*100:.2f}%.')
    
def get_predict_args():
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
    parser.add_argument('image_path', type=str, help='The path to flower image file', default='flowers/test/77/image_00005.jpg')
    parser.add_argument('checkpoint', type=str, help='The checkpoint', default='checkpoint/checkpoint.pth')
    parser.add_argument('--top_k', type=int, help='Top k classes', default=5)
    parser.add_argument('--category_names', type=str, help='The category names file path', default='cat_to_name.json')
    parser.add_argument('--gpu', action="store_true", default=True)
    
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    args = parser.parse_args()
    
    return args