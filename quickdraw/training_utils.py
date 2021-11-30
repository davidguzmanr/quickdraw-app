import random
import numpy as np
from tqdm import trange, tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import models
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

def train_epoch(model, train_loader, optimizer, device):
    """
    Trains a single epoch.

    model: PyTorch model
        Model to train.

    train_loader: DataLoader.
        PyTorch dataloader with the training data.

    optimizer: torch.optim
        Optmizer to train the model.

    device: torch.device.
        Device where the model will be trained, 'cpu' or 'gpu'.
    """
    model.train()
    for x, y_true in tqdm(train_loader, leave=False):
        # Move to GPU, in case there is one
        x, y_true = x.to(device), y_true.to(device)
        
        # Compute logits
        y_lgts = model(x)
        
        # Compute the loss
        loss = F.cross_entropy(y_lgts, y_true)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def eval_epoch(model, data_loader, device):
    """
    Evaluates a single epoch.
    
    model: PyTorch model.
        Model to train.

    data_loader: DataLoader.
        PyTorch dataloader with data to validate.

    device: torch.device.
        Device where the model will be trained, 'cpu' or 'gpu'.
    """
    model.eval()
    with torch.no_grad():
        losses, accs = [], []

        for x, y_true in tqdm(data_loader, leave=False):
            # Move to GPU, in case there is one
            x, y_true = x.to(device), y_true.to(device)

            # Compute logits
            y_lgts = model(x)

            # Compute scores
            y_prob = F.softmax(y_lgts, dim=1)

            # Get the classes
            y_pred = torch.argmax(y_prob, dim=1)

            # Compute the loss
            loss = F.cross_entropy(y_lgts, y_true)

            # Compute accuracy
            accuracy = (y_true == y_pred).type(torch.float32).mean()

            # Save the current loss and accuracy
            losses.append(loss.item())
            accs.append(accuracy.item())

        # Compute the mean
        loss = np.mean(losses) * 100
        accuracy = np.mean(accs) * 100

        return loss, accuracy

def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Saves a checkpoint of the model for the current epoch.

    model: PyTorch model.
        Model to train.

    optimizer: PyTorch optimizer.
        Optimizer for the model.    

    epoch: int.
        Number of the current epoch.

    loss: float.
        Loss (in train) of the current epoch.

    path: str.
        Path to save the checkpoint.
    """
    if path:
        torch.save(
            {'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(), 
             'loss': loss,
             }, path)

def train(model, train_loader, validation_loader, device, lr=1e-3, epochs=20, patience=5, 
          writer=None, checkpoint_path=None):
    """
    Trains the whole model.

    model: PyTorch model.
        Model to train.

    train_loader: DataLoader.
        PyTorch dataloader with the training data.

    validation_loader: DataLoader.
        PyTorch dataloader with the validation data.

    device: torch.device
        Device where the model will be trained, 'cpu' or 'gpu'

    lr: float, default=1e-3.
        Learning rate.
    
    epochs: int, default=20.
        Number of epochs.

    patience: int, default=5.
        Number of epochs with no improvement after which training 
        will be stopped for early stopping.

    writer: instance of SummaryWriter, default=None.
        Writer for Tensorboard.

    checkpoint_path: str, default=None.
        Path to save checkpoints for the model, if None no checkpoints 
        will be saved.

    Notes
    -----
    - See https://clay-atlas.com/us/blog/2021/08/25/pytorch-en-early-stopping/
    """
    last_loss = np.inf
    early_stop = 0
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in trange(epochs):
        # Train a single epoch
        train_epoch(model, train_loader, optimizer, device)

        # Evaluate in training
        train_loss, train_acc = eval_epoch(model, train_loader, device)
        # Evaluate in validation
        val_loss, val_acc = eval_epoch(model, validation_loader, device)

        if writer:
            writer.add_scalar(tag='Loss/train', scalar_value=train_loss, global_step=epoch)
            writer.add_scalar(tag='Accuracy/train', scalar_value=train_acc, global_step=epoch)

            writer.add_scalar(tag='Loss/validation', scalar_value=val_loss, global_step=epoch)
            writer.add_scalar(tag='Accuracy/validation', scalar_value=val_acc, global_step=epoch)

        # Early stopping
        current_loss = val_loss
        if current_loss > last_loss:
            early_stop += 1
            if early_stop > patience:
                print('Early stopping!')
                return # Stop training
        else:
            early_stop = 0
            save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)

        last_loss = current_loss

def create_model():
    """
    Creates a MobileNet_V2 model and replaces the classifier layer for this task.
    """
    model = models.mobilenet_v2()
    # Replace the last layer so out_features=345 instead of out_features=1000
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=1280, out_features=345)
    )

    return model

def load_dataset():
    """
    Create a PyTorch Dataset for the images.

    Notes
    -----
    - See https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949 
    """
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.9720, 0.9720, 0.9720), 
                             (0.1559, 0.1559, 0.1559)) # Normalize with the mean and std of the whole dataset
    ])

    dataset = ImageFolder(root='images', transform=transform)

    return dataset