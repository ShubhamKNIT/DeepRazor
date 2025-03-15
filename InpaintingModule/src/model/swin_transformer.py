import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from timm.models.swin_transformer import SwinTransformerBlock

def interpolate_relative_position_bias_table(old_bias, new_num):
    """
    Interpolates a relative position bias table from the old size to the new size.
    
    Parameters:
      old_bias: tensor of shape (old_num, num_heads)
      new_num: desired total number of entries (should be a perfect square, e.g. 225 for window_size=8, because (2*8-1)^2 = 15^2 = 225)
    
    Returns:
      new_bias: tensor of shape (new_num, num_heads)
    """
    old_num, num_heads = old_bias.shape  # e.g., (169, num_heads)
    old_size = int(math.sqrt(old_num))    # should be 13 for 7x7
    new_size = int(math.sqrt(new_num))    # should be 15 for 8x8
    # Reshape to (1, num_heads, old_size, old_size)
    old_bias_4d = old_bias.transpose(0, 1).contiguous().view(1, num_heads, old_size, old_size)
    # Interpolate to new size (1, num_heads, new_size, new_size)
    new_bias_4d = F.interpolate(old_bias_4d, size=(new_size, new_size), mode='bicubic', align_corners=False)
    # Reshape back to (new_num, num_heads)
    new_bias = new_bias_4d.view(num_heads, new_size * new_size).transpose(0, 1)
    return new_bias


class SwinBottleneckTransformer(nn.Module):
    """
    A Swin-style transformer block for bottleneck feature refinement.
    It partitions the input feature map into non-overlapping windows of size (ws, ws),
    processes each window with SwinTransformerBlock(s), and then reconstructs the full feature map.
    
    Expected input: (B, embed_dim, H, W) with H and W divisible by window_size.
    Output: (B, embed_dim, H, W)
    """
    def __init__(self, embed_dim=128, num_heads=4, window_size=8, num_layers=1, mlp_ratio=4.0):
        super(SwinBottleneckTransformer, self).__init__()
        self.window_size = window_size
        ws = window_size
        # Initialize a stack of SwinTransformerBlocks.
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=embed_dim,
                input_resolution=(ws, ws),
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio
            )
            for _ in range(num_layers)
        ])
        # In your __init__ method after loading pretrained_model:
        pretrained_model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        # Access the first block of the first layer
        pretrained_block_state = pretrained_model.layers[0].blocks[0].state_dict()
        # Adapt the relative position bias table
        for k in pretrained_block_state:
            if "attn.relative_position_bias_table" in k:
                old_bias = pretrained_block_state[k]
                # For window_size=8, we expect (2*8-1)^2 = 15^2 = 225 entries
                new_bias = interpolate_relative_position_bias_table(old_bias, new_num=(2*8-1)**2)
                pretrained_block_state[k] = new_bias
                print(f"Interpolated {k} from {old_bias.shape} to {new_bias.shape}")
                
        for blk in self.blocks:
            try:
                blk.load_state_dict(pretrained_block_state)
            except Exception as e:
                print("State dict load failed:", e)

    def forward(self, x):
        # x: (B, embed_dim, H, W)
        B, C, H, W = x.shape
        ws = self.window_size
        assert H % ws == 0 and W % ws == 0, "H and W must be divisible by window_size."
        # Partition x into windows.
        x_windows = x.view(B, C, H // ws, ws, W // ws, ws)
        x_windows = x_windows.permute(0, 2, 4, 3, 5, 1).contiguous()
        num_windows = B * (H // ws) * (W // ws)
        x_windows = x_windows.view(num_windows, ws, ws, C)
        # Process each window with transformer blocks.
        for blk in self.blocks:
            x_windows = blk(x_windows)
        # Reshape back to full feature map.
        x_windows = x_windows.view(B, H // ws, W // ws, ws, ws, C)
        x_windows = x_windows.permute(0, 5, 1, 3, 2, 4).contiguous()
        out = x_windows.view(B, C, H, W)
        return out

# Standalone test
if __name__ == "__main__":
    dummy_input = torch.randn(2, 128, 32, 32)
    print("Input shape:", dummy_input.shape)
    model = SwinBottleneckTransformer(embed_dim=128, num_heads=4, window_size=8, num_layers=1, mlp_ratio=4.0)
    output = model(dummy_input)
    print("Output shape:", output.shape)