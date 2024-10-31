import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from transformers import ViTForImageClassification
from dataset import BrainTumorDataset
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import confusion_matrix
import os

import torch


class TransformerModel:
    def __init__(self, batch_size=32):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        self.dataset = BrainTumorDataset(transform=transform)
        self.train_loader, self.test_loader, self.val_loader = None, None, None

        # device config | CUDA if GPU avail.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Pre trained model
        self.model = ViTForImageClassification.from_pretrained(
            pretrained_model_name_or_path='google/vit-base-patch16-224',
            num_labels=4,
            ignore_mismatched_sizes=True
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-5)
        self.criterion = nn.CrossEntropyLoss()

        self.train_losses, self.train_accuracies, self.val_losses, self.val_accuracies = [], [], [], []

        # epochs used during training
        self.epochs: int = 0

        self._prepare_data(batch_size=batch_size)

    def _prepare_data(self, train_split=0.7, val_split=0.1, batch_size=32):
        train_size = int(train_split * len(self.dataset))
        val_size = int(val_split * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(self.dataset, [train_size, val_size, test_size])

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return self.train_loader, self.test_loader, self.val_loader

    def train_epoch(self):
        # Set model to train mode
        self.model.train()

        total_loss, correct = 0, 0
        for images, labels in tqdm(self.train_loader, desc='Training', leave=False):
            images, labels = images.to(self.device), labels.to(self.device)

            # Reset optimizer
            self.optimizer.zero_grad()

            # Forward
            outputs = self.model(images).logits
            loss = self.criterion(outputs, labels)

            # backward
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()

        accuracy = correct / len(self.train_loader.dataset)
        return total_loss / len(self.train_loader), accuracy

    def evaluate(self, loader):
        # Set model to eval mode
        self.model.eval()

        total_loss, correct = 0, 0
        with torch.no_grad():
            for images, labels in tqdm(loader, desc='Evaluating', leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images).logits
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
            accuracy = correct / len(loader.dataset)
            return total_loss / len(loader), accuracy

    def train(self, epochs=10):
        for epoch in range(epochs):
            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy = self.evaluate(self.val_loader)

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            self.epochs = epoch + 1

    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(self.epochs), self.train_losses, label='Train Loss')
        plt.plot(range(self.epochs), self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title("Training and Validation Loss Over Epochs")
        plt.show()

    def plot_accuracy(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(self.epochs), self.train_accuracies, label='Train Accuracy')
        plt.plot(range(self.epochs), self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

    def plot_confusion_matrix(self):
        all_preds = []
        all_labels = []

        # Set model to eval mode
        self.model.eval()

        # Collect predictions and true labels
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Confusion Matrix Generation', leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images).logits
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        class_labels = self.dataset.classes

        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.show()

    def test(self):
        test_loss, test_accuracy = self.evaluate(self.test_loader)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    def save(self, model_name):
        path = os.path.join(os.getcwd(), 'saved_models', model_name)
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(self.model.state_dict, path)

    def predict(self, image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
        image = Image.open(image_path).convert("RGB")

        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)

        with torch.no_grad():
            outputs = self.model(image).logits
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        probabilities = probabilities.cpu().numpy().flatten()

        for label, prob in zip(self.dataset.classes, probabilities):
            print(f"{label}: {prob * 100:.2f}%")
