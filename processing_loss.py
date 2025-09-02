import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss.
    This loss function computes the SSIM between two images, which is a measure of the similarity between them.
    """
    def __init__(self, L, K1=0.01, K2=0.03):
        """
        Initialize the SSIM loss function.

        Args:
            L: The dynamic range of the pixel values (typically 255 for 8-bit images).
            K1: The first constant for stability (default: 0.01).
            K2: The second constant for stability (default: 0.03).
        """
        super(SSIMLoss, self).__init__()
        self.c1 = (K1 * L) ** 2
        self.c2 = (K2 * L) ** 2
        
    def forward(self, x, y):
        """
        Compute the SSIM loss between two images.

        Args:
            x: The first input image (batch_size, channels, height, width)
            y: The second input image (batch_size, channels, height, width)

        Returns:
            The SSIM loss value (scalar).
        """
        mu_x = torch.mean(x, dim=(2, 3), keepdim=True)
        mu_y = torch.mean(y, dim=(2, 3), keepdim=True)
        
        sigma_x = torch.var(x, dim=(2, 3), keepdim=True, unbiased=False)
        sigma_y = torch.var(y, dim=(2, 3), keepdim=True)
        
        sigma_xy = torch.mean((x - mu_x) * (y - mu_y), dim=(2, 3), keepdim=True)
        
        ssim = ((2 * mu_x * mu_y + self.c1) * (2 * sigma_xy + self.c2)) / ((mu_x ** 2 + mu_y ** 2 + self.c1) * (sigma_x ** 2 + sigma_y ** 2 + self.c2))
        
        return 1 - torch.mean(ssim)
    
class FourierLoss(nn.Module):
    """
    Fourier loss.
    This loss function computes the Fourier transform of the input images and compares them in the frequency domain.
    """
    def __init__(self):
        super(FourierLoss, self).__init__()
        
    def forward(self, x, y):
        """
        Compute the Fourier loss between two images.

        Args:
            x: The first input image (batch_size, channels, height, width)
            y: The second input image (batch_size, channels, height, width)

        Returns:
            The Fourier loss value (scalar).
        """
        x_fft = torch.fft.fft2(x, norm='ortho')
        y_fft = torch.fft.fft2(y, norm='ortho')
        
        loss = F.l1_loss(x_fft, y_fft)
        
        return loss
    
class ProcessingLoss(nn.Module):
    """
    Processing loss.
    This loss function combines multiple loss components to guide the image processing.
    """
    def __init__(self, L, lambda1=1.0, lambda2=0.5, lambda3=0.2):
        """
        Initialize the Processing loss function.

        Args:
            L: The dynamic range of the pixel values (typically 255 for 8-bit images).
            lambda1: Weight for the MAE loss component (default: 1.0).
            lambda2: Weight for the SSIM loss component (default: 0.5).
            lambda3: Weight for the Fourier loss component (default: 0.2).
        """
        super(ProcessingLoss, self).__init__()
        self.ssim_loss = SSIMLoss(L=L)
        self.fourier_loss = FourierLoss()
        
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        
    def forward(self, x, y):
        """
        Compute the forward pass of the Processing loss.

        Args:
            x: The first input image (batch_size, channels, height, width)
            y: The second input image (batch_size, channels, height, width)

        Returns:
            The total processing loss value (scalar).
        """
        mae = F.l1_loss(x, y)
        ssim = self.ssim_loss(x, y)
        fourier = self.fourier_loss(x, y)
    
        total_loss = (self.lambda1 * mae) + (self.lambda2 * ssim) + (self.lambda3 * fourier)
        
        return total_loss 
            
