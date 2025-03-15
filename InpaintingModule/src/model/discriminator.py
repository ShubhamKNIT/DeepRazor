from src.model.conv_layers import *

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.sn = True
        self.norm = 'in'
        self.activation = 'lrelu'

        self.conv_layer_1 = Conv2dLayer(4, 64, 3, 2, 1, 1, sn=False, norm=self.norm, activation=self.activation)
        self.conv_layer_2 = Conv2dLayer(64, 128, 3, 2, 1, 1, sn=self.sn, norm=self.norm, activation=self.activation)
        self.conv_layer_3 = Conv2dLayer(128, 256, 3, 2, 1, 1, sn=self.sn, norm=self.norm, activation=self.activation)
        self.conv_layer_4 = Conv2dLayer(256, 256, 3, 2, 1, 1, sn=self.sn, norm=self.norm, activation=self.activation)
        self.conv_layer_5 = Conv2dLayer(256, 1, 3, 2, 1, 1, sn=self.sn, norm=self.norm, activation=self.activation)

    def forward(self, img, mask):
        x = img * mask
        x = torch.cat([x, mask], dim=1)
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_4(x)
        x = self.conv_layer_5(x)
        return x