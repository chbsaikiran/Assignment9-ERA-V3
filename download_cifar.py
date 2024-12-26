import torchvision
import os
import shutil
from torchvision.datasets import CIFAR10

def setup_cifar10():
    # Download CIFAR10
    trainset = CIFAR10(root='./cifar10', train=True, download=True)
    testset = CIFAR10(root='./cifar10', train=False, download=True)
    
    # Create directories
    for split in ['train', 'val']:
        for class_idx in range(10):
            os.makedirs(f'data/{split}/{class_idx}', exist_ok=True)
    
    # Move images to the correct directories
    for i, (img, label) in enumerate(trainset):
        img.save(f'data/train/{label}/img_{i}.png')
    
    for i, (img, label) in enumerate(testset):
        img.save(f'data/val/{label}/img_{i}.png')

if __name__ == '__main__':
    setup_cifar10() 