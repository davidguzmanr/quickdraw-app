First create the dataset to train the model, to do that just run 

```bash
python create_dataset.py
```

I am just using 1,000 images per class but there are more images in the original dataset (see
[The Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset)). 

To train the model run

```bash
python training.py --batch-size=32 --epochs=50 --lr=1e-3 --cuda
```

To see the progress in Tensorboard

```bash
tensorboard --logdir=runs
```