from quickdraw import QuickDrawDataGroup
from tqdm import tqdm
import os

def main():
    """
    Download the images and create the necessary directories to store them.

    Notes
    -----
    - See https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.ImageFolder to see
      how images must be arranged for the DataLoader.
    """
    if not os.path.exists('images'):
        os.mkdir('images')

    with open('categories/categories.txt') as file:
        names = [name.replace('\n', '') for name in file.readlines()]

    for name in tqdm(names):
        images = QuickDrawDataGroup(name, recognized=True, max_drawings=1000, 
                                    cache_dir='bin-images', print_messages=False)
        name = name.replace(' ', '-')
        path = f'images/{name}/'

        if not os.path.exists(path):
            os.mkdir(path)

        for drawing in images.drawings:
            drawing.image.save(f'images/{name}/{drawing.key_id}.jpg')

if __name__ == '__main__':
    main()