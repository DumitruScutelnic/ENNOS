import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from .pro_net import ProNet
from ENNOS.utils import load_model

class DoubleConv(nn.Module):
    """
    A double convolutional block consisting of two convolutional layers with ReLU activation.
    It is used in the UNet architecture.
    """
    def __init__(self, input_shape: int, output_shape: int) -> None:
        """
        Args:
            input_shape (int): Number of input channels.
            output_shape (int): Number of output channels.
        """
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(input_shape, output_shape, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_shape, output_shape, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)
    
class DownSample(nn.Module):
    """
    A downsampling block consisting of a double convolutional layer followed by a max pooling layer.
    It is used in the UNet architecture to reduce the spatial dimensions of the input.
    """
    def __init__(self, input_shape: int, output_shape: int) -> None:
        """
        Args:
            input_shape (int): Number of input channels.
            output_shape (int): Number of output channels.
        """
        super().__init__()
        self.conv = DoubleConv(input_shape, output_shape)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p
    
class UpSample(nn.Module):
    """
    An upsampling block consisting of a transposed convolutional layer followed by a double convolutional layer.
    It is used in the UNet architecture to increase the spatial dimensions of the input.
    """
    def __init__(self, input_shape: int, output_shape: int ) -> None:
        """
        Args:
            input_shape (int): Number of input channels.
            output_shape (int): Number of output channels.
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(input_shape, input_shape//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(input_shape, output_shape)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)
    
class UNet(nn.Module):
    """
    A UNet architecture for image segmentation tasks.
    It consists of a series of downsampling and upsampling blocks.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        """
        Args:
            input_shape (int): Number of input channels.
            hidden_units (int): Number of hidden units.
            output_shape (int): Number of output channels.
        """
        super().__init__()
        self.down_convolution_1 = DownSample(input_shape, hidden_units)
        self.down_convolution_2 = DownSample(hidden_units, hidden_units*2)
        self.down_convolution_3 = DownSample(hidden_units*2, hidden_units*4)
        self.down_convolution_4 = DownSample(hidden_units*4, hidden_units*8)

        self.bottle_neck = DoubleConv(hidden_units*8, hidden_units*16)

        self.up_convolution_1 = UpSample(hidden_units*16, hidden_units*8)
        self.up_convolution_2 = UpSample(hidden_units*8, hidden_units*4)
        self.up_convolution_3 = UpSample(hidden_units*4, hidden_units*2)
        self.up_convolution_4 = UpSample(hidden_units*2, hidden_units)

        self.out = nn.Conv2d(in_channels=hidden_units, out_channels=output_shape, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out
    
class ENNOSModel(nn.Module):
    """
    A model that combines the PreNet and UNet architectures.
    It first processes the input through the PreNet model that gives a preprocessed image,
    and then passes the output through the UNet for segmentation tasks.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, device: torch.device, model_name: str) -> None:
        """
        Args:
            input_shape (int): Number of input channels.
            hidden_units (int): Number of hidden units.
            output_shape (int): Number of output channels.
            device (torch.device): The device to run the model on (CPU or GPU).
            weights_path (str): Path to the pre-trained weights for the PreNet.
        """
        super(ENNOSModel, self).__init__()
        self.prenet = ProNet(input_shape, hidden_units, output_shape).to(device)
        self.unet = UNet(input_shape, hidden_units, output_shape).to(device)
      
        load_model(model=self.prenet, model_name=model_name, device=device)
        # freeze the PreNet model parameters
        for param in self.prenet.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        x = self.prenet(x)
        x = self.unet(x)
        return x