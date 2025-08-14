import gymnasium as gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import timm
import numpy as np
class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    A custom feature extractor that uses a pre-trained EfficientNet-B3 for
    image features and an MLP for vector data.
    It is modified to handle stacked frames.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim)

        image_space = observation_space["image"]
        vector_space = observation_space["vector"]
        
        # The number of input channels for the CNN is now 3 * num_stacked_frames
        n_input_channels = image_space.shape[-1] 

        # --- Image Feature Extractor (Pre-trained EfficientNet) ---
        # We load a pre-trained EfficientNet, but we will modify its first layer
        # to accept the stacked frames (e.g., 12 channels instead of 3).
        self.cnn = timm.create_model('efficientnet_b3', pretrained=True, in_chans=n_input_channels)
        
        # We only want the feature extraction layers, not the final classifier
        self.cnn.classifier = nn.Identity()

        # Freeze the weights of the pre-trained model (except for the first conv layer)
        for name, param in self.cnn.named_parameters():
            if 'conv_stem' not in name: # The first convolution layer is named 'conv_stem'
                param.requires_grad = False
            else:
                # We need to train the new first layer
                param.requires_grad = True

        # Determine the output feature size of the CNN
        with torch.no_grad():
            sample_image = torch.as_tensor(image_space.sample()[None]).float()
            # Permute from (N, H, W, C) to (N, C, H, W)
            sample_image = sample_image.permute(0, 3, 1, 2)
            n_flatten = self.cnn(sample_image).shape[1]

        # --- Vector Feature Extractor (MLP) ---
        # The vector input is also stacked, so we flatten it
        vector_input_dim = np.prod(vector_space.shape)
        self.mlp = nn.Sequential(
            nn.Linear(vector_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # --- Combined Feature Layer ---
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + 64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict) -> torch.Tensor:
        # The observation from FrameStack is a LazyFrames object, convert to tensor
        image_obs = torch.as_tensor(observations["image"], device=self.device).float()
        vector_obs = torch.as_tensor(observations["vector"], device=self.device).float()

        # Permute image channels for PyTorch (N, H, W, C) -> (N, C, H, W)
        image_tensor = image_obs.permute(0, 3, 1, 2)
        
        # Flatten the stacked vector observations
        vector_input_dim = np.prod(vector_obs.shape[1:])
        vector_tensor = vector_obs.view(-1, vector_input_dim)

        # Pass inputs through their respective networks
        cnn_features = self.cnn(image_tensor)
        vector_features = self.mlp(vector_tensor)

        # Concatenate features and pass through the final linear layer
        combined_features = torch.cat((cnn_features, vector_features), dim=1)
        return self.linear(combined_features)