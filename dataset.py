import kagglehub
import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np


class Dataset:
    def __init__(self):
        self.path = kagglehub.dataset_download("rm1000/brain-tumor-mri-scans")
        self.classes = ['glioma', 'healthy', 'meningioma', 'pituitary']
        self.classes_dirs = [os.path.join(self.path, c) for c in self.classes]

    def plot_random_images(self, num_images=10):
        for idx, d in enumerate(self.classes_dirs):
            image_files = [os.path.join(d, f) for f in os.listdir(d)]
            random_images = random.sample(image_files, num_images)

            plt.figure(figsize=(10, 10))
            for i, img_path in enumerate(random_images):
                image = cv2.imread(img_path, 0)  # images are grayscale, so we use 0 or cv2.IMREAD_GRAYSCALE
                plt.imshow(image, cmap='gray')
                plt.axis('off')
                print(i)
            print(f"Random image sample for {self.classes[idx]}")
            plt.show()

    # Image histogram tells the frequency of pixel intensities, in this case 0 - 256
    def plot_histograms_for_images(self, num_images=10):
        for idx, d in enumerate(self.classes_dirs):
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
        for idx, d in enumerate(self.classes_dirs):
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
        return self.path

    def get_classes(self):
        return self.classes

    def get_classes_dirs(self):
        return self.classes_dirs
