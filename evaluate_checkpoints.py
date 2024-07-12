import os
import torch
import torch.nn as nn
import argparse
from utils import get_network, get_test_dataloader
from conf import settings

def eval_model(checkpoint_path, net, test_loader, device):
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net.eval()
    correct = 0
    total = 0
    loss_function = nn.CrossEntropyLoss()
    test_loss = 0.0

    with torch.no_grad():
        for (labels, images) in test_loader:
            labels = torch.LongTensor(list(labels))
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = loss_function(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    average_loss = test_loss / len(test_loader.dataset)
    return accuracy, average_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    net = get_network(args).to(device)
    test_path = "data/etl_952_singlechar_size_64/952_test"
    test_loader = get_test_dataloader(
        path=test_path,
        num_workers=4,
        batch_size=128,
        shuffle=False
    )

    # Modify this part to match your directory structure
    checkpoint_dir = os.path.join(settings.CHECKPOINT_PATH, args.net)
    subdir = "Thursday_11_July_2024_18h_42m_34s"
    checkpoint_subdir = os.path.join(checkpoint_dir, subdir)
    
    accuracies = []

    for epoch in range(1, settings.EPOCH + 1):
        checkpoint_path = os.path.join(checkpoint_subdir, f'{args.net}-{epoch}-regular.pth')
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            accuracy, avg_loss = eval_model(checkpoint_path, net, test_loader, device)
            print(f"Epoch: {epoch}, Accuracy: {accuracy:.2f}%, Average Loss: {avg_loss:.4f}")
            accuracies.append((epoch, accuracy, avg_loss))
        else:
            print(f"Checkpoint {checkpoint_path} does not exist")

    # Save the accuracies to a file
    with open('accuracies.txt', 'w') as f:
        for epoch, accuracy, avg_loss in accuracies:
            f.write(f"Epoch: {epoch}, Accuracy: {accuracy:.2f}%, Average Loss: {avg_loss:.4f}\n")

if __name__ == '__main__':
    main()
