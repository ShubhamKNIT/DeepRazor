import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTMAEModel

class MATInpainting(nn.Module):
    def __init__(self, pretrained_model='facebook/vit-mae-base'):
        super(MATInpainting, self).__init__()
        # Load the pretrained ViT-MAE model
        self.mae = ViTMAEModel.from_pretrained(pretrained_model)
        
        # Define a simple CNN decoder (ViT-MAE outputs features with hidden dim 768)
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output pixel values in [0,1]
        )
    
    def forward(self, x, mask):
        """
        Args:
            x: Input image tensor of shape [B, 3, H, W] (expected H=W=512 in your case).
            mask: Binary mask tensor of shape [B, 1, H, W] (currently not directly used).
        """
        # Resize the input to 224x224 as expected by the pretrained model
        x_resized = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        
        # Forward pass through the ViT-MAE encoder
        outputs = self.mae(pixel_values=x_resized)
        latent = outputs.last_hidden_state  # Shape: [B, num_patches, 768]
        latent = latent[:, 1:, :]            # now shape: [B, 49, 768]

        # Assume num_patches forms a perfect square; reshape into a spatial feature map
        b, n, c = latent.shape
        patch_dim = int(n ** 0.5)
        latent_map = latent.permute(0, 2, 1).contiguous().view(b, c, patch_dim, patch_dim)

        # Decode the latent feature map to reconstruct the image
        decoded = self.decoder(latent_map)  # This output will be at the transformer resolution
        
        # Upsample the output back to the original image size (e.g., 512x512)
        out_up = F.interpolate(decoded, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out_up