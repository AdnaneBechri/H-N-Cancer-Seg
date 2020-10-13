"""
Source

All the code here is based on these Github repositories :

https://github.com/UdonDa/3D-UNet-PyTorch

https://github.com/josedolz/LiviaNET

https://github.com/josedolz/HyperDenseNet

"""
# 3D-UNet model.
# x: 128x128 resolution for 32 frames.
import torch
import torch.nn as nn


def conv_block_out(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),)


def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation, )


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        activation, )


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation, )


class UNet_FED(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(UNet_FED, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)

        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_4 = max_pooling_3d()
        self.down_5 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
        self.pool_5 = max_pooling_3d()

        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 16, self.num_filters * 32, activation)

        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 96, self.num_filters * 32, activation)
        self.up_1 = conv_block_2_3d(self.num_filters * 48, self.num_filters * 16, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.up_2 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation)

        # Output
        #self.out = conv_block_out(self.num_filters, out_dim)
        self.out = conv_block_3d(self.num_filters, out_dim, activation)

    def forward(self, x):
        # Down sampling
        down_11 = self.down_1(x[:,0:1,:,:,:])  # -> [1, 4, 128, 128, 128]
        pool_11 = self.pool_1(down_11)  # -> [1, 4, 64, 64, 64]

        down_12 = self.down_1(x[:,1:2,:,:,:])  # -> [1, 4, 128, 128, 128]
        pool_12 = self.pool_1(down_12)  # -> [1, 4, 64, 64, 64]

        down_13 = self.down_1(x[:, 2:3, :, :, :])  # -> [1, 4, 128, 128, 128]
        pool_13 = self.pool_1(down_13)  # -> [1, 4, 64, 64, 64]


        #catf = torch.cat((pool_11, pool_12), 1)

        down_21 = self.down_2(pool_11)  # -> [1, 8, 64, 64, 64]
        pool_21 = self.pool_2(down_21)  # -> [1, 8, 32, 32, 32]

        down_31 = self.down_3(pool_21)  # -> [1, 16, 32, 32, 32]
        pool_31 = self.pool_3(down_31)  # -> [1, 16, 16, 16, 16]

        down_41 = self.down_4(pool_31)  # -> [1, 32, 16, 16, 16]
        pool_41 = self.pool_4(down_41)  # -> [1, 32, 8, 8, 8]

        down_51 = self.down_5(pool_41)  # -> [1, 64, 8, 8, 8]
        pool_51 = self.pool_5(down_51)  # -> [1, 64, 4, 4, 4]
        # -------------------
        down_22 = self.down_2(pool_12)  # -> [1, 8, 64, 64, 64]
        pool_22 = self.pool_2(down_22)  # -> [1, 8, 32, 32, 32]

        down_32 = self.down_3(pool_22)  # -> [1, 16, 32, 32, 32]
        pool_32 = self.pool_3(down_32)  # -> [1, 16, 16, 16, 16]

        down_42 = self.down_4(pool_32)  # -> [1, 32, 16, 16, 16]
        pool_42 = self.pool_4(down_42)  # -> [1, 32, 8, 8, 8]

        down_52 = self.down_5(pool_42)  # -> [1, 64, 8, 8, 8]
        pool_52 = self.pool_5(down_52)  # -> [1, 64, 4, 4, 4]
        # ---------------------
        down_23 = self.down_2(pool_13)  # -> [1, 8, 64, 64, 64]
        pool_23 = self.pool_2(down_23)  # -> [1, 8, 32, 32, 32]

        down_33 = self.down_3(pool_23)  # -> [1, 16, 32, 32, 32]
        pool_33 = self.pool_3(down_33)  # -> [1, 16, 16, 16, 16]

        down_43 = self.down_4(pool_33)  # -> [1, 32, 16, 16, 16]
        pool_43 = self.pool_4(down_43)  # -> [1, 32, 8, 8, 8]

        down_53 = self.down_5(pool_43)  # -> [1, 64, 8, 8, 8]
        pool_53 = self.pool_5(down_53)  # -> [1, 64, 4, 4, 4]


        # Bridge
        bridge1 = self.bridge(pool_51)  # -> [1, 128, 4, 4, 4]
        bridge2 = self.bridge(pool_52)  # -> [1, 128, 4, 4, 4]
        bridge3 = self.bridge(pool_53)  # -> [1, 128, 4, 4, 4]
        bridge  = torch.cat([bridge1, bridge2, bridge3], dim=1)
        # Up sampling

        trans_1 = self.trans_1(bridge)  # -> [1, 128, 8, 8, 8]
        concat_1 = torch.cat([trans_1, torch.add(torch.add(down_51,down_52), down_53)], dim=1)  # -> [1, 192, 8, 8, 8]
        up_1 = self.up_1(concat_1)  # -> [1, 64, 8, 8, 8]

        trans_2 = self.trans_2(up_1)  # -> [1, 64, 16, 16, 16]
        concat_2 = torch.cat([trans_2, torch.add(torch.add(down_41,down_42), down_43)], dim=1)  # -> [1, 96, 16, 16, 16]
        up_2 = self.up_2(concat_2)  # -> [1, 32, 16, 16, 16]

        trans_3 = self.trans_3(up_2)  # -> [1, 32, 32, 32, 32]
        concat_3 = torch.cat([trans_3, torch.add(torch.add(down_31,down_32), down_33)], dim=1)  # -> [1, 48, 32, 32, 32]
        up_3 = self.up_3(concat_3)  # -> [1, 16, 32, 32, 32]

        trans_4 = self.trans_4(up_3)  # -> [1, 16, 64, 64, 64]
        concat_4 = torch.cat([trans_4, torch.add(torch.add(down_21,down_22), down_23)], dim=1)  # -> [1, 24, 64, 64, 64]
        up_4 = self.up_4(concat_4)  # -> [1, 8, 64, 64, 64]

        trans_5 = self.trans_5(up_4)  # -> [1, 8, 128, 128, 128]
        concat_5 = torch.cat([trans_5, torch.add(torch.add(down_11,down_12), down_13)], dim=1)  # -> [1, 12, 128, 128, 128]
        up_5 = self.up_5(concat_5)  # -> [1, 4, 128, 128, 128]

        # Output
        out = self.out(up_5)  # -> [1, 3, 128, 128, 128]
        return out


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 128
    x = torch.Tensor(1, 3, image_size, image_size, image_size)
    x.to(device)
    print("x size: {}".format(x.size()))

    model = UNet(in_dim=3, out_dim=3, num_filters=4)

    out = model(x)
    print("out size: {}".format(out.size()))