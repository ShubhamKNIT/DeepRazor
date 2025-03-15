from src.model.coarse import *
from src.model.refine import *

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # GatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type='replicate', activation=elu, norm=none, sc=false, sn=false)
        self.coarse_generator = Coarse()
        self.refine_generator = Refine()

    def forward(self, img, mask):
        # coarse inpainting
        img_256 = F.interpolate(img, size=[256, 256], mode='bilinear')
        mask_256 = F.interpolate(mask, size=[256, 256], mode='nearest')
        masked_img_256 = img_256 * (1 - mask_256) + mask_256
        coarse_img = self.coarse_generator(torch.cat([masked_img_256, mask_256], dim=1))
        coarse_img = F.interpolate(coarse_img, size=[512, 512], mode='bilinear')

        # coarse replaces hole in original image and serve as input
        img_coarse = img * (1 - mask) + coarse_img * mask
        inpainted_img = self.refine_generator(img_coarse, mask)

        return coarse_img, inpainted_img