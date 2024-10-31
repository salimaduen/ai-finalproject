import kagglehub
import os
import random
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from preprocessing import *


class BrainTumorDataset(Dataset):
    def __init__(self, transform=None):
        self._path = kagglehub.dataset_download("rm1000/brain-tumor-mri-scans")
        self.classes = ['glioma', 'healthy', 'meningioma', 'pituitary']
        self._classes_dirs = [os.path.join(self._path, c) for c in self.classes]

        self.images, self.labels, self.images_dict = self._load_images()

        self.transform = transform

    def _init_images_dict(self):
        images_dict = dict()
        for i in range(len(self.classes)):
            images_dict[i] = []
        return images_dict

    def _load_images(self):
        images, labels = [], []
        # Structured dict format {label: [list of images], . . .}
        images_dict = self._init_images_dict()
        for idx, class_dir in enumerate(self._classes_dirs):
            image_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir)]
            for img_path in image_files:
                image = cv2.imread(img_path, 0)  # images are grayscale, so we use 0 or cv2.IMREAD_GRAYSCALE
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    image = resize_image(image)
                    images.append(image)
                    labels.append(idx)
                    images_dict.get(idx).append(image)
        return np.array(images), np.array(labels), images_dict

    def plot_random_images(self, num_images=10):
        for label, images in self.images_dict.items():
            random_images = random.sample(images, num_images)

            fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
            for img, ax in zip(random_images, axes):
                ax.imshow(img, cmap='gray')
                ax.axis('off')
            print(f"Random image sample for {self.classes[label]}")
            plt.show()

    # Image histogram tells the frequency of pixel intensities, in this case 0 - 256
    def plot_histograms_for_images(self, num_images=10):
        for idx, d in enumerate(self._classes_dirs):
            image_files = [os.path.join(d, f) for f in os.listdir(d)]
            random_images = random.sample(image_files, num_images)

            plt.figure(figsize=(10, 5))
            for img_path in random_images:
                image = cv2.imread(img_path, 0)
                hist = cv2.calcHist([image], [0], None, [256], [0, 256])
                plt.plot(hist)

            plt.title(f'Pixel Intensity Histogram for Multiple Images ({self.classes[idx]})')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.show()

    def calculate_intensity_stats(self, num_images=100):
        for idx, d in enumerate(self._classes_dirs):
            image_files = [os.path.join(d, f) for f in os.listdir(d)]
            random_images = random.sample(image_files, num_images)

            means, stds = [], []
            for img_path in random_images:
                image = cv2.imread(img_path, 0)
                means.append(np.mean(image))
                stds.append(np.std(image))

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.hist(means, bins=50)
            plt.title(f'Mean Pixel Intensity Distribution ({self.classes[idx]})')
            plt.xlabel('Mean Intensity')
            plt.ylabel('Frequency')

            plt.subplot(1, 2, 2)
            plt.hist(stds, bins=50)
            plt.title(f'Standard Deviation of Intensity Distribution ({self.classes[idx]})')
            plt.xlabel('Standard Deviation')
            plt.ylabel('Frequency')

            plt.show()

    def get_path(self):
        return self._path

    def get_classes(self):
        return self.classes

    def get_classes_dirs(self):
        return self._classes_dirs

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        label = self.labels[idx]

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label
