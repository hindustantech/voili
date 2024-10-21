import clip
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image


class Model:
    def __init__(self, settings_path: str = './settings.yaml'):
        # Load settings from YAML file
        with open(settings_path, "r") as file:
            self.settings = yaml.safe_load(file)

        # Ensure necessary settings are available
        try:
            self.device = self.settings['model-settings'].get('device', 'cpu')
            self.model_name = self.settings['model-settings'].get('model-name', 'ViT-B/32')  # Default to a smaller model
            self.threshold = self.settings['model-settings'].get('prediction-threshold', 0.5)
            self.labels = self.settings['label-settings']['labels']
            self.default_label = self.settings['label-settings'].get('default-label', 'Unknown')
        except KeyError as e:
            raise ValueError(f"Missing key in settings: {e}")

        # Load the CLIP model
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        
        # Prepare labels for the model
        self.labels_ = [f'a photo of {label}' for label in self.labels]
        self.text_features = self.vectorize_text(self.labels_)

    @torch.no_grad()
    def transform_image(self, image: np.ndarray):
        pil_image = Image.fromarray(image).convert('RGB')
        tf_image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        return tf_image

    @torch.no_grad()
    def tokenize(self, text: list):
        text = clip.tokenize(text).to(self.device)
        return text

    @torch.no_grad()
    def vectorize_text(self, text: list):
        tokens = self.tokenize(text=text)
        text_features = self.model.encode_text(tokens)
        return text_features

    @torch.no_grad()
    def predict_(self, text_features: torch.Tensor, image_features: torch.Tensor):
        # Normalize the features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T
        values, indices = similarity[0].topk(1)
        return values, indices

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> dict:
        '''
        Does prediction on an input image.

        Args:
            image (np.ndarray): Numpy image with RGB channel ordering type.
                              Convert image to RGB if you read via OpenCV.

        Returns:
            dict: Contains predictions with 'label' and 'confidence'.
        '''
        tf_image = self.transform_image(image)
        image_features = self.model.encode_image(tf_image)
        values, indices = self.predict_(text_features=self.text_features, image_features=image_features)
        
        label_index = indices[0].cpu().item()
        model_confidence = abs(values[0].cpu().item())
        
        # Check confidence against the threshold
        label_text = self.default_label
        if model_confidence >= self.threshold:
            label_text = self.labels[label_index]

        prediction = {
            'label': label_text,
            'confidence': model_confidence
        }

        return prediction

    @staticmethod
    def plot_image(image: np.ndarray, title_text: str):
        plt.figure(figsize=[13, 13])
        plt.title(title_text)
        plt.axis('off')
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.show()  # Ensure the image is displayed
