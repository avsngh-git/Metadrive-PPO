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
    
    MODIFIED to work correctly with stable-baselines3's automatic wrappers.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim)

        image_space = observation_space["image"]
        vector_space = observation_space["vector"]
        
        # --- MODIFICATION: The input channels are now the *first* dimension ---
        # The VecTransposeImage wrapper has already moved the channel dimension.
        n_input_channels = image_space.shape[0]

        # --- Image Feature Extractor (Pre-trained EfficientNet) ---
        self.cnn = timm.create_model('efficientnet_b3', pretrained=True, in_chans=n_input_channels)
        self.cnn.classifier = nn.Identity()

        # Freeze the weights of the pre-trained model
        for name, param in self.cnn.named_parameters():
            if 'conv_stem' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Determine the output feature size of the CNN
        with torch.no_grad():
            # Create a sample tensor that matches the PyTorch channel-first format
            sample_image = torch.as_tensor(image_space.sample()[None]).float()
            n_flatten = self.cnn(sample_image).shape[1]

        # --- Vector Feature Extractor (MLP) ---
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
        image_obs = torch.as_tensor(observations["image"], device=self.device).float()
        vector_obs = torch.as_tensor(observations["vector"], device=self.device).float()

        # --- MODIFICATION: The .permute() call has been REMOVED ---
        # The input is now in the correct (N, C, H, W) format.
        
        # Flatten the stacked vector observations
        vector_input_dim = np.prod(vector_obs.shape[1:])
        vector_tensor = vector_obs.view(-1, vector_input_dim)

        # Pass inputs through their respective networks
        cnn_features = self.cnn(image_obs)
        vector_features = self.mlp(vector_tensor)

        # Concatenate features and pass through the final linear layer
        combined_features = torch.cat((cnn_features, vector_features), dim=1)
        return self.linear(combined_features)