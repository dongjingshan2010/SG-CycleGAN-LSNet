import torch.nn as nn
import torch.nn.functional as F
from DRconv import DRconv2d

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class SharedGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, shared_layers=4):
        super(SharedGenerator, self).__init__()

        # 共享层
        shared_model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            # DRconv2d(in_channels=input_nc, out_channels=64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # 下采样层
        in_features = 64
        out_features = in_features * 2
        for i in range(2):
            shared_model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

            # 控制共享层数（初始卷积层+两层下采样=4层）
            if i + 1 >= shared_layers - 1:
                break

        self.shared_layers = nn.Sequential(*shared_model)

        # A2B私有层
        a2b_private_model = []
        a2b_in_features = in_features

        a2b_private_model += [DRconv2d(in_channels=a2b_in_features, out_channels=a2b_in_features, kernel_size=1),]

        # 完成下采样（如果有未共享的下采样层）
        if shared_layers <= 3:
            for i in range(2 - (shared_layers - 1)):
                a2b_private_model += [
                    nn.Conv2d(a2b_in_features, a2b_in_features * 2, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(a2b_in_features * 2),
                    nn.ReLU(inplace=True)
                ]
                a2b_in_features = a2b_in_features * 2

        # A2B残差块
        for _ in range(n_residual_blocks):
            a2b_private_model += [ResidualBlock(a2b_in_features)]

        # A2B上采样
        a2b_out_features = a2b_in_features // 2
        for _ in range(2):
            a2b_private_model += [
                nn.ConvTranspose2d(a2b_in_features, a2b_out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(a2b_out_features),
                nn.ReLU(inplace=True)
            ]
            a2b_in_features = a2b_out_features
            a2b_out_features = a2b_in_features // 2

        # A2B输出层
        a2b_private_model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()
        ]

        self.a2b_private_layers = nn.Sequential(*a2b_private_model)

        # B2A私有层
        b2a_private_model = []
        b2a_in_features = in_features

        # 完成下采样（如果有未共享的下采样层）
        if shared_layers <= 3:
            for i in range(2 - (shared_layers - 1)):
                b2a_private_model += [
                    nn.Conv2d(b2a_in_features, b2a_in_features * 2, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(b2a_in_features * 2),
                    nn.ReLU(inplace=True)
                ]
                b2a_in_features = b2a_in_features * 2

        # B2A残差块
        for _ in range(n_residual_blocks):
            b2a_private_model += [ResidualBlock(b2a_in_features)]

        # B2A上采样
        b2a_out_features = b2a_in_features // 2
        for _ in range(2):
            b2a_private_model += [
                nn.ConvTranspose2d(b2a_in_features, b2a_out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(b2a_out_features),
                nn.ReLU(inplace=True)
            ]
            b2a_in_features = b2a_out_features
            b2a_out_features = b2a_in_features // 2

        # B2A输出层
        b2a_private_model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, input_nc, 7),
            nn.Tanh()
        ]

        self.b2a_private_layers = nn.Sequential(*b2a_private_model)

    def forward_a2b(self, x):
        x = self.shared_layers(x)
        return self.a2b_private_layers(x), x

    def forward_b2a(self, x):
        x = self.shared_layers(x)
        return self.b2a_private_layers(x), x


