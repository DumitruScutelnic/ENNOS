import torch.nn as nn

class ResNetBlock(nn.Module):
    """
    A residual block for the ResNet architecture.
    It consists of two convolutional layers with batch normalization and LeakyReLU activation.
    The input is added to the output of the block (skip connection).
    """
    def __init__(self, input_shape: int, output_shape: int) -> None:
        """
        Args:
            input_shape (int): Number of input channels.
            output_shape (int): Number of output channels.
        """
        super(ResNetBlock, self).__init__()
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=output_shape, kernel_size=1),
            nn.BatchNorm2d(output_shape)
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=output_shape, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_shape),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=output_shape, out_channels=output_shape, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_shape),
            nn.LeakyReLU()
        )

    def forward(self, x):
        identity = self.skip(x)
        x = self.conv_block(x)
        x += identity # skip connection
        return x

class ProNet(nn.Module):
    """
    A model that consists of a series of convolutional layers and residual blocks.
    It is inspired by the Super-Resolution Residual Network (SRResNet) architecture.
    "Ledig et al., Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    <https://arxiv.org/pdf/1609.04802>"
    The model is designed to learn a preprocessing pipeline for images.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        """
        Args:
            input_shape (int): Number of input channels.
            hidden_units (int): Number of hidden units.
            output_shape (int): Number of output channels.
        """
        super(ProNet, self).__init__()
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(hidden_units),
            nn.LeakyReLU()
        )

        self.down = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_units * 2),
            nn.LeakyReLU()
        )

        self.res_blocks = nn.Sequential(*[ResNetBlock(hidden_units * 2, hidden_units * 2) for _ in range(4)])

        self.up = nn.Sequential(
            nn.ConvTranspose2d(hidden_units * 2, hidden_units, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.LeakyReLU()
        )
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(hidden_units),
            nn.LeakyReLU()
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=output_shape, kernel_size=9, stride=1, padding=4),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv_1(x)
        residual = x
        x = self.down(x)
        x = self.res_blocks(x)
        x = self.up(x)
        x = self.conv_2(x)
        x += residual # skip connection
        x = self.out(x)
        return x