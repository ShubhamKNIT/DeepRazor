from src.model.gated_conv_layers import *

class Coarse(nn.Module):
    def __init__(self):
        super(Coarse, self).__init__()
        # Initialize the padding scheme
        self.coarse1 = nn.Sequential(
            # encoder
            GatedConv2d(4, 32, 5, 2, 2, sc=True),
            GatedConv2d(32, 32, 3, 1, 1, sc=True),
            GatedConv2d(32, 64, 3, 2, 1, sc=True)
        )
        self.coarse2 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 1, sc=True),
            GatedConv2d(64, 64, 3, 1, 1, sc=True),
            GatedConv2d(64, 64, 3, 1, 1, sc=True)
        )
        self.coarse3 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 1, sc=True),
            GatedConv2d(64, 64, 3, 1, 1, sc=True),
            GatedConv2d(64, 64, 3, 1, 1, sc=True)
        )
        self.coarse4 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 2, dilation=2, sc=True),
            GatedConv2d(64, 64, 3, 1, 2, dilation=2, sc=True),
            GatedConv2d(64, 64, 3, 1, 2, dilation=2, sc=True)
        )
        self.coarse5 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 4, dilation=4, sc=True),
            GatedConv2d(64, 64, 3, 1, 4, dilation=4, sc=True),
            GatedConv2d(64, 64, 3, 1, 4, dilation=4, sc=True)
        )
        self.coarse6 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 8, dilation=8, sc=True),
            GatedConv2d(64, 64, 3, 1, 8, dilation=8, sc=True),
            GatedConv2d(64, 64, 3, 1, 8, dilation=8, sc=True)
        )
        self.coarse7 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 16, dilation=16, sc=True),
            GatedConv2d(64, 64, 3, 1, 16, dilation=16, sc=True)
        )
        self.coarse8 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 1, sc=True),
            GatedConv2d(64, 64, 3, 1, 1, sc=True),
            GatedConv2d(64, 64, 3, 1, 1, sc=True),
        )
        # decoder
        self.coarse9 = nn.Sequential(
            TransposeGatedConv2d(64, 64, 3, 1, 1, sc=True),
            TransposeGatedConv2d(64, 32, 3, 1, 1, sc=True),
            GatedConv2d(32, 3, 3, 1, 1, activation='none', sc=True),
            nn.Tanh()
        )

    def forward(self, first_in):
        first_out = self.coarse1(first_in)
        first_out = self.coarse2(first_out) + first_out
        first_out = self.coarse3(first_out) + first_out
        first_out = self.coarse4(first_out) + first_out
        first_out = self.coarse5(first_out) + first_out
        first_out = self.coarse6(first_out) + first_out
        first_out = self.coarse7(first_out) + first_out
        first_out = self.coarse8(first_out) + first_out
        first_out = self.coarse9(first_out)
        first_out = torch.clamp(first_out, 0, 1)
        return first_out