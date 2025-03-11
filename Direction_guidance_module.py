import torch
import torch.nn as nn
import torch.nn.functional as F
from Direction_convolution import DC  # Import DC from the direction_convolution module

class DGM(nn.Module):
    r"""
    Args:
    dim (int): Number of input channels.
    input_resolution(tuple[int]): Input resolution.
    """
    def __init__(self, dim, input_resolution):
        super(DGM, self).__init__()
        self.dc = DC(input_resolution)  # Directional Convolution
        self.proj_q = nn.Conv2d(12, 12 * 2 * 2, kernel_size=2, stride=2)  # Project query (Q)
        self.proj_k = nn.Conv2d(12, 12 * 2 * 2, kernel_size=2, stride=2)  # Project key (K)
        self.proj_v = nn.Conv2d(dim, dim * 2 * 2, kernel_size=2, stride=2)  # Project value (V)
        self.softmax = nn.Softmax(dim=-1)  # Softmax for attention computation

    def forward(self, x):
        b = x.size()[0]  # Batch size
        c = x.size()[1]  # Number of channels
        h, w = x.size(2), x.size(3)  # Height and Width of the feature map

        # Apply Directional Convolution (DC)
        dir_map = self.dc(x)

        # Project Q, K, V
        q = self.proj_q(dir_map).view(b, 12 * 2 * 2, -1)  # Reshape Q for attention
        k = self.proj_k(dir_map).view(b, 12 * 2 * 2, -1)  # Reshape K for attention
        v = self.proj_v(x).view(b, c * 2 * 2, -1)  # Reshape V for attention

        # Attention computation
        attention = self.softmax(torch.bmm(q.permute(0, 2, 1), k))  # Compute attention scores
        output = torch.bmm(v, attention.permute(0, 2, 1))  # Apply attention to values
        
        return output
