import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from training_utils import load_dataset, create_model, train, eval_epoch
import argparse

def main():
    # See https://github.com/pytorch/examples/blob/master/mnist/main.py 
    # for a nice example for the ArgumentParser

    # Training settings
    parser = argparse.ArgumentParser(description='QuickDraw training')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='Number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--cuda', action='store_true', dest='cuda',
                        help='Enables CUDA training.')
    parser.add_argument('--no-cuda', action='store_false', dest='cuda',
                        help='Disables CUDA training.')
    parser.set_defaults(cuda=True)
    args = parser.parse_args()

    # If there is a GPU and CUDA is enabled the model will be trained in the GPU
    if torch.cuda.is_available() and args.cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(f'\nTraining in {device}\n')

    # Create model and move it to the device
    model = create_model()
    model.to(device)
    summary(model)
    
    # Create the dataset and split it in (train, val, test), I will use (0.8, 0.1, 0.1)
    dataset = load_dataset()  
    n = len(dataset)  
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset, 
        lengths=[int(0.8 * n), int(0.1 * n), int(0.1 * n)], 
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        num_workers=4
    )
    validation_loader = DataLoader(
        validation_dataset, 
        batch_size=args.batch_size, 
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        num_workers=4
    )

    writer = SummaryWriter(log_dir='runs/experiment-2')
    
    # Train the model, open TensorBoard to see the progress
    train(
        model, 
        train_loader, 
        validation_loader, 
        device, 
        lr=args.lr, 
        epochs=args.epochs, 
        writer=writer, 
        checkpoint_path='models/checkpoint-2.pt'
    )

    # Save the model
    torch.save(model, 'models/model-2.pt')

    # Add some metrics to evaluate different models and hyperparameters
    _, train_acc = eval_epoch(model, train_loader, device)
    _, val_acc = eval_epoch(model, validation_loader, device)
    _, test_acc = eval_epoch(model, test_loader, device)

    writer.add_hparams(
        hparam_dict={
            'lr': args.lr, 
            'batch_size': args.batch_size, 
            'epochs': args.epochs
        },
        metric_dict={
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc
        },
        run_name='hparams'
    )

if __name__ == '__main__':
    main()