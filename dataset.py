import os
import numpy as np
import torch
import csv
from torch.utils.data import Dataset
# from skimage import io
from PIL import Image
from torchvision import transforms
from collections import Counter
# import matplotlib.pyplot as plt

def read_labels(label_file):
    labels = {}
    with open(label_file, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        next(reader, None)  # Skip the header
        for line in reader:
            label = int(line[0])
            raw = line[1]
            sequence = line[-1]
            # Treat "zc" as a separate character
            sequence = sequence.replace('zc', '/')
            labels[label] = (raw, sequence)
    
    # Print the labels dictionary for debugging
    # print("Labels dictionary:", labels)
    
    return labels

def build_vocab(labels):
    char_counter = Counter()
    for label, (raw, sequence) in labels.items():
        char_counter.update(sequence)
    
    vocab = {char: count for char, count in char_counter.items()}
    
    # Print the vocabulary for debugging
    print("Vocabulary:", vocab)
    
    return vocab


class CustomDataset(Dataset):
    def __init__(self, root_dir, label_file, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            label_file (string): Path to the label file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root_dir) for f in filenames if 'png' in f]
        
        # Read the label file into a dictionary
        self.labels = read_labels(label_file)
        
        # Build the vocabulary from the sequences
        self.vocab = build_vocab(self.labels)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_name = self.image_files[index]
        label = int(os.path.basename(os.path.dirname(img_name)))  # Extract the label from the parent directory name
        image = Image.open(img_name)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        raw, sequence = self.labels[label]
        
        # Tokenize the sequence using the vocabulary
        tokens = [self.vocab[char] for char in sequence]
        sequence = torch.tensor(tokens, dtype=torch.long)

        return label, image, sequence, raw
    
# Example usage
if __name__ == "__main__":
    root_dir = '/shared/data/etl_952_singlechar_size_64/952_train'
    label_file = '/shared/data/etl_952_singlechar_size_64/952_labels.txt'
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = CustomDataset(root_dir=root_dir, label_file=label_file, transform=transform)
    
    # Test a specific image by index
    specific_index = 7000  # Change this to the index of the specific image you want to test
    label, image, sequence_indices, raw = dataset[specific_index]
    print(f"Label: {label}, Sequence Indices: {sequence_indices}, Raw: {raw}")