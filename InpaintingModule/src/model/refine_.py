from src.model.cam import *
from src.model.gated_conv_layers import *

class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()

        self.cam = CAM()

        # downsample
        self.ref_block_1 = nn.Sequential(
            GatedConv2d(3, 32, 5, 2, 2, 1),   # (H/2, W/2)
            GatedConv2d(32, 32, 3, 1, 1, 1)  # (H/2, W/2)
        )
        self.ref_block_2 = nn.Sequential(
            GatedConv2d(32, 64, 3, 2, 1, 1),  # (H/4, W/4)
            GatedConv2d(64, 64, 3, 1, 1, 1)  # (H/4, W/4)
        )
        self.ref_block_3 = nn.Sequential(
            GatedConv2d(64, 128, 3, 2, 1, 1),  # (H/8, W/8)
        )

        # dilated block (used for feature-extraction and attention score calculation)
        self.dilated_block_1 = nn.Sequential(
            GatedConv2d(128, 128, 3, 1, 1, 1),  # (H/8, W/8)
            GatedConv2d(128, 128, 3, 1, 1, 1)  # (H/8, W/8)
        )
        self.dilated_block_2 = nn.Sequential(
            GatedConv2d(128, 128, 3, 1, 2, 2),  # (H/8, W/8)
            GatedConv2d(128, 128, 3, 1, 4, 4)  # (H/8, W/8)
        )
        self.dilated_block_3 = nn.Sequential(
            GatedConv2d(128, 128, 3, 1, 8, 8),  # (H/8, W/8)
            GatedConv2d(128, 128, 3, 1, 16, 16) # (H/8, W/8)
        )

        # upsample
        self.transposed_ref_block_3 = nn.Sequential(
            GatedConv2d(256, 128, 3, 1, 1, 1),
            TransposeGatedConv2d(128, 64, 3, 1, 1, 1),
            GatedConv2d(64, 64, 3, 1, 1, 1)
        )
        self.transposed_ref_block_2 = nn.Sequential(
            TransposeGatedConv2d(128, 64, 3, 1, 1, 1),
            GatedConv2d(64, 32, 3, 1, 1, 1)
        )
        self.transposed_ref_block_1 = nn.Sequential(
            TransposeGatedConv2d(64, 32, 3, 1, 1, 1),
            GatedConv2d(32, 3, 3, 1, 1, 1, activation='none'),
            nn.Tanh()
        )

        # extract low-level features
        self.llf_pl_3 = nn.Sequential(   # input taken from ref_block3
            GatedConv2d(128, 128, 3, 1, 1, 1),
        )
        self.llf_pl_2 = nn.Sequential(   # input taken from ref_block2
            GatedConv2d(64, 64, 3, 1, 1, 1),
            GatedConv2d(64, 64, 3, 1, 2, 2),
        )
        self.llf_pl_1 = nn.Sequential(   # input taken from ref_block1
            GatedConv2d(32, 32, 3, 1, 1, 1),
            GatedConv2d(32, 32, 3, 1, 2, 2),
        )

    def forward(self, img_coarse, mask):
        # encoder: downsample
        pl1 = self.ref_block_1(img_coarse)
        pl2 = self.ref_block_2(pl1)
        llf = self.ref_block_3(pl2)

        # calculate attention scores from low-levelfeatures
        llf = self.dilated_block_1(llf) + llf
        llf = self.dilated_block_2(llf) + llf
        pl3 = self.dilated_block_3(llf) + llf

        patch_fb = self.cam.calculate_patches(32, mask, 512)    # get patches
        attention = self.cam.compute_attention(pl3, patch_fb)   # calculate attention scores

        # decoder: upsample
        residulal_pl3 = self.cam.attention_transfer(pl3, attention)
        trans_pl3 = torch.cat([pl3, self.llf_pl_3(residulal_pl3)], dim=1)
        trans_pl3 = self.transposed_ref_block_3(trans_pl3)

        residual_pl2 = self.cam.attention_transfer(pl2, attention)
        trans_pl2 = torch.cat([trans_pl3, self.llf_pl_2(residual_pl2)], dim=1)
        trans_pl2 = self.transposed_ref_block_2(trans_pl2)

        residual_pl1 = self.cam.attention_transfer(pl1, attention)
        trans_pl1 = torch.cat([trans_pl2, self.llf_pl_1(residual_pl1)], dim=1)
        trans_pl1 = self.transposed_ref_block_1(trans_pl1)

        inpainted_img = torch.clamp(trans_pl1, 0, 1)
        return inpainted_img