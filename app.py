import gradio as gr
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms

with open('quickdraw/categories/id_to_class.json') as file:
    id_to_class = json.load(file)

# Define the model and load the weights
model = models.mobilenet_v2()
model.classifier = nn.Sequential(nn.Dropout(p=0.2),
                                nn.Linear(in_features=1280, out_features=345))
checkpoint = torch.load('quickdraw/models/model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use the same preprocessing used or training
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.9720, 0.9720, 0.9720), 
                         (0.1559, 0.1559, 0.1559)) # Normalize with the mean and std of the whole dataset
])

def predict_drawing(image):
    """
    Predict the drawing using the trained model.
    """
    image = transform(image)
    image = torch.unsqueeze(image, dim=0) # Add batch dimension
    output = model(image)
    probabilities = F.softmax(output[0], dim=0)

    classes = {id_to_class[str(i)]: prob.item() for (i, prob) in enumerate(probabilities)}

    return classes

image = gr.inputs.Image(shape=(255, 255), source='canvas', type='numpy')

iface = gr.Interface(
    fn=predict_drawing,
    inputs=image,
    outputs=gr.outputs.Label(num_top_classes=3),
    live=False,
    interpretation='default'
)

if __name__ == '__main__':
    iface.launch()
