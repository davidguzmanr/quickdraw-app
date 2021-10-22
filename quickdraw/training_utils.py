import random
import numpy as np
from tqdm import trange, tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

def train_epoch(model, train_loader, optimizer, device):
    """
    Train a single epoch.

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
    for x, y_true in tqdm(train_loader):
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
    Evaluate a single epoch.
    
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

        for x, y_true in data_loader:
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

def save_checkpoint(model, epoch, path):
    """
    Save a checkpoint of the model for the current epoch.

    model: PyTorch model.
        Model to train.

    epoch: int.
        Number of the current epoch.

    path: str.
        Path to save the checkpoint.
    """
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict()}, 
                path)

def train(model, train_loader, validation_loader, device, lr=1e-3, epochs=20, patience=5, 
          train_writer=None, validation_writer=None):
    """
    Train the whole model.

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

    train_writer: instance of SummaryWriter, default=None.
        Writer for the training dataset in Tensorboard.

    validation_writer: instance of SummaryWriter, default=None.
        Writer for the validation dataset in Tensorboard.

    patience: int, default=5
        Number of epochs with no improvement after which training 
        will be stopped for early stopping.
    """
    loss_hist, acc_hist = [], []
    last_loss = np.inf
    early_stop = 0
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in trange(epochs):
        # Train a single epoch
        train_epoch(model, train_loader, optimizer, device)

        # Evaluate in training
        train_loss, train_acc = eval_epoch(model, train_loader, device)
        # Evaluate in validation
        val_loss, val_acc = eval_epoch(model, validation_loader, device)

        loss_hist.append([train_loss, train_acc])
        acc_hist.append([val_loss, val_acc])

        if train_writer:
            train_writer.add_scalar(tag='metrics/loss', scalar_value=train_loss, global_step=epoch)
            train_writer.add_scalar(tag='metrics/accuracy', scalar_value=train_acc, global_step=epoch)

        if validation_writer:
            validation_writer.add_scalar(tag='metrics/loss', scalar_value=val_loss, global_step=epoch)
            validation_writer.add_scalar(tag='metrics/accuracy', scalar_value=val_acc, global_step=epoch)

        # Early stopping
        current_loss = val_loss
        if current_loss > last_loss:
            early_stop += 1
            if early_stop > patience:
                print('Early stopping!')
                return loss_hist, acc_hist
        else:
            early_stop = 0
            save_checkpoint(model, epoch, 'models/model.pt')

        last_loss = current_loss

    return loss_hist, acc_hist