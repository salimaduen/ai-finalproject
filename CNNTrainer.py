import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from dataset import BrainTumorDataset
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from sklearn.metrics import confusion_matrix


class Trainer:
    def __init__(self, model, batch_size=32):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)

        self.dataset = BrainTumorDataset(transform=transform)
        self.train_loader, self.test_loader, self.val_loader = None, None, None

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss()

        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)

        self._prepare_data(batch_size=batch_size)

        self.train_losses, self.train_accuracies, self.val_losses, self.val_accuracies = [], [], [], []

        self.epochs = 0

    def _prepare_data(self, train_split=0.7, val_split=0.1, batch_size=32):
        train_size = int(train_split * len(self.dataset))
        val_size = int(val_split * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(self.dataset, [train_size, val_size, test_size])

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return self.train_loader, self.test_loader, self.val_loader

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            train_loss, correct = 0, 0

            for images, labels in tqdm(self.train_loader, desc='Training', leave=False):
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()

            self.epochs += 1
            self.scheduler.step()
            train_accuracy = correct / len(self.train_loader.dataset)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            # Validate after each epoch
            self.validate()

    def validate(self):
        self.model.eval()  # Set the model to evaluation mode
        val_loss, correct = 0, 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Evaluating', leave=False):
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()

        val_accuracy = correct / len(self.val_loader.dataset)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    def test(self):
        self.model.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Evaluating', leave=False):
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Metrics
                test_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()

        test_accuracy = correct / len(self.test_loader.dataset)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

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
                outputs = self.model(images)
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

    def predict(self, image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

        try:
            # Open and preprocess the image
            image = Image.open(image_path).convert("RGB")
            image = transform(image)
            image = image.unsqueeze(0).to(self.device)

            self.model.eval()

            # Perform inference
            with torch.no_grad():
                outputs = self.model(image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probabilities = probabilities.cpu().numpy().flatten()

            for label, prob in zip(self.dataset.classes, probabilities):
                print(f"{label}: {prob * 100:.2f}%")

        except FileNotFoundError:
            print(f"Error: File not found at path '{image_path}'")
        except Exception as e:
            print(f"An error occurred: {e}")
