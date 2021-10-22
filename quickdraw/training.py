import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from torchvision.datasets import ImageFolder
from torchvision import models
import torchvision.transforms as transforms

from training_utils import train

def main():
    # If there is a GPU then the model will be trained there
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = models.mobilenet_v2()
    # Replace the last layer so out_features=345 instead of out_features=1000
    model.classifier = nn.Sequential(nn.Dropout(p=0.2),
                                     nn.Linear(in_features=1280, out_features=345))
    # Move model to GPU, in case there is one
    model.to(device)

    # See https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.9720, 0.9720, 0.9720), 
                             (0.1559, 0.1559, 0.1559)) # Normalize with the mean and std of the whole dataset
    ])

    dataset = ImageFolder(root='images', transform=transform)
    train_dataset, validation_dataset = random_split(dataset, lengths=[30000, 4500],
                                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, 
                              generator=torch.Generator().manual_seed(42))
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True, num_workers=4, 
                                   generator=torch.Generator().manual_seed(42))

    train_writer = SummaryWriter(log_dir='runs/train-3')
    validation_writer = SummaryWriter(log_dir='runs/validation-3')

    _, _ = train(model, train_loader, validation_loader, device, lr=1e-3, epochs=20, 
                 train_writer=train_writer, validation_writer=validation_writer)

if __name__ == '__main__':
    main()