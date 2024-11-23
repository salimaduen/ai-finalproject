import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        # Hook to capture gradients
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activation = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def compute_heatmap(self, input_tensor, target_class):
        self.model.eval()
        output = self.model(input_tensor)

        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()

        gradients = self.gradients.cpu().data.numpy()
        activations = self.activation.cpu().data.numpy()

        weights = np.mean(gradients, axis=(2, 3))[0]

        cam = np.zeros(activations.shape[2:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[0, i]

        # Normalize heatmap
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam
