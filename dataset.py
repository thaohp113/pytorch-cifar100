import os
import numpy as np
import torch
import csv
from torch.utils.data import Dataset
# from skimage import io
from PIL import Image
from torchvision import transforms
from collections import Counter
# from utils import read_labels, build_vocab
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
    # print("Vocabulary:", vocab)
    
    return vocab


class CustomDataset(Dataset):
    def __init__(self, root_dir, label_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root_dir) for f in filenames if 'png' in f]
        
        self.labels = read_labels(label_file)
        self.vocab = build_vocab(self.labels)
        self.max_seq_length = max(len(seq) for _, seq in self.labels.values())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_name = self.image_files[index]
        label = int(os.path.basename(os.path.dirname(img_name)))
        image = Image.open(img_name)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        raw, sequence = self.labels[label]
        tokens = [self.vocab[char] for char in sequence]
        
        # Pad sequence
        padded_sequence = torch.zeros(self.max_seq_length, dtype=torch.long)
        padded_sequence[:len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        
        sequence_length = torch.tensor(len(tokens), dtype=torch.long)

        return label, image, padded_sequence, sequence_length, raw
    
# # Example usage
# if __name__ == "__main__":
#     root_dir = '/shared/data/etl_952_singlechar_size_64/952_train'
#     label_file = '/shared/data/etl_952_singlechar_size_64/952_labels.txt'
    
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
    
#     dataset = CustomDataset(root_dir=root_dir, label_file=label_file, transform=transform)
    
#     specific_index = 7000
#     label, image, sequence_indices, sequence_length, raw = dataset[specific_index]
#     print(f"Label: {label}, Sequence Indices: {sequence_indices}, Sequence Length: {sequence_length}, Raw: {raw}")