import torch.nn as nn
import torch.nn.functional as F
import torch
from directional_convolution import DGM

# Swish activation function
def swish(x):
    return x * F.relu6(x + 3) / 6


# Basic convolution block with options for batch normalization and activation function
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, bias=False, bn=True,
                 out='dis', activation=swish, conv=nn.Conv2d,
                 ):
        super(BasicConv2d, self).__init__()

        # Calculate padding based on kernel size and dilation
        if not isinstance(kernel_size, tuple):
            if dilation > 1:
                padding = dilation * (kernel_size // 2)  # AtrousConv2d
            elif kernel_size == stride:
                padding = 0
            else:
                padding = kernel_size // 2  # BasicConv2d

        # Convolution layer
        self.c = conv(in_channels, out_channels,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=bias)

        # Batch normalization (if enabled)
        self.b = nn.BatchNorm2d(out_channels, momentum=0.01) if bn else nn.Identity()

        # Dropout layer for regularization
        self.o = nn.Identity()
        drop_prob = 0.15
        self.o = nn.Dropout2d(p=drop_prob, inplace=False)

        # Set activation function (default is swish)
        if activation is None:
            self.a = nn.Identity()
        else:
            self.a = activation

    def forward(self, x):
        x = self.c(x)  # Apply convolution
        x = self.o(x)  # Apply dropout
        x = self.b(x)  # Apply batch normalization
        x = self.a(x)  # Apply activation function
        return x


# Bottleneck module used for deeper models
class Bottleneck(nn.Module):
    MyConv = BasicConv2d

    def __init__(self, in_c, out_c, stride=1, downsample=None, **args):
        super(Bottleneck, self).__init__()

        # First convolution layer
        self.conv1 = self.MyConv(in_c, out_c, kernel_size=3, stride=stride)

        # Second convolution layer (standard convolution + batch normalization)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c)
        )

        # Activation function (Swish)
        self.relu = swish

        # Downsampling layer (if needed for skip connection)
        if downsample is None and in_c != out_c:
            downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_c),
            )
        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)  # Apply downsample (if needed)

        # Apply convolutions and activation
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.relu(out + residual)  # Add skip connection
        return out


# Standard convolution block with options for pooling, shortcut connections
class ConvBlock(torch.nn.Module):
    attention = None
    MyConv = BasicConv2d

    def __init__(self, inp_c, out_c, ksize=3, shortcut=False, pool=True):
        super(ConvBlock, self).__init__()

        # Shortcut connection (1x1 convolution with batch normalization)
        self.shortcut = nn.Sequential(nn.Conv2d(inp_c, out_c, kernel_size=1), nn.BatchNorm2d(out_c))

        # Max pooling layer (if enabled)
        if pool:
            self.pool = nn.MaxPool2d(kernel_size=2)
        else:
            self.pool = False

        # Add two convolutions to form the block
        block = []
        block.append(self.MyConv(inp_c, out_c, kernel_size=ksize, padding=(ksize - 1) // 2))
        block.append(self.MyConv(out_c, out_c, kernel_size=ksize, padding=(ksize - 1) // 2))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        if self.pool:
            x = self.pool(x)  # Apply pooling if enabled
        out = self.block(x)  # Apply convolutions
        return swish(out + self.shortcut(x))  # Add shortcut connection and apply activation


# Upsample block for upscaling feature maps
class UpsampleBlock(torch.nn.Module):
    def __init__(self, inp_c, out_c, up_mode='transp_conv'):
        super(UpsampleBlock, self).__init__()
        block = []

        # Select upsampling method (either transposed convolution or bilinear interpolation)
        if up_mode == 'transp_conv':
            block.append(nn.ConvTranspose2d(inp_c, out_c, kernel_size=2, stride=2))
        elif up_mode == 'up_conv':
            block.append(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False))
            block.append(nn.Conv2d(inp_c, out_c, kernel_size=1))
        else:
            raise Exception('Upsampling mode not supported')

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)  # Apply upsampling
        return out


# Convolution bridge block to refine features
class ConvBridgeBlock(torch.nn.Module):
    def __init__(self, out_c, ksize=3):
        super(ConvBridgeBlock, self).__init__()
        block = []

        # Standard convolution block with LeakyReLU and BatchNorm
        block.append(nn.Conv2d(out_c, out_c, kernel_size=ksize, padding=(ksize - 1) // 2))
        block.append(nn.LeakyReLU())
        block.append(nn.BatchNorm2d(out_c))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)  # Apply convolution and activation
        return out


# Up-convolution block for upsampling and concatenation with skip connections
class UpConvBlock(torch.nn.Module):
    def __init__(self, inp_c, out_c, ksize=3, up_mode='up_conv', conv_bridge=False, shortcut=False):
        super(UpConvBlock, self).__init__()
        self.conv_bridge = conv_bridge

        # Upsample the feature maps
        self.up_layer = UpsampleBlock(inp_c, out_c, up_mode=up_mode)
        # Convolution block to refine the features after concatenation
        self.conv_layer = ConvBlock(2 * out_c, out_c, ksize=ksize, shortcut=shortcut, pool=False)

        # Add a convolution bridge if enabled
        if self.conv_bridge:
            self.conv_bridge_layer = ConvBridgeBlock(out_c, ksize=ksize)

    def forward(self, x, skip):
        up = self.up_layer(x)  # Upsample the input feature maps
        if self.conv_bridge:
            out = torch.cat([up, self.conv_bridge_layer(skip)], dim=1)  # Concatenate with skip connection (via bridge)
        else:
            out = torch.cat([up, skip], dim=1)  # Concatenate with skip connection
        out = self.conv_layer(out)  # Apply convolution to the concatenated features
        return out


# DGNet model
class DGNet(nn.Module):

    def __init__(self, inp_c, input_resolution, n_classes=1, layers=(32, 32, 32, 32, 32)):
        super(DGNet, self).__init__()

        # Set the number of output features (the last layer size)
        self.num_features = layers[-1]

        # Define the model name based on the number of layers and their size
        self.__name__ = 'u{}x{}'.format(len(layers), layers[0])
        self.n_classes = n_classes

        # The first convolutional layer
        self.first = BasicConv2d(inp_c, layers[0])

        # Downsampling path (encoder)
        self.down_path = nn.ModuleList()
        for i in range(len(layers) - 1):
            # Add each convolution block to the downsampling path
            block = ConvBlock(inp_c=layers[i], out_c=layers[i + 1], pool=True)
            self.down_path.append(block)

        # Upsampling path (decoder)
        self.up_path = nn.ModuleList()
        reversed_layers = list(reversed(layers))
        for i in range(len(layers) - 1):
            # Add each upsampling block to the upsampling path
            block = UpConvBlock(inp_c=reversed_layers[i], out_c=reversed_layers[i + 1])
            self.up_path.append(block)

        # Convolution and batch normalization after the upsampling path
        self.conv_bn = nn.Sequential(
            nn.Conv2d(layers[0], layers[0], kernel_size=1),  # 1x1 convolution to refine features
            nn.BatchNorm2d(layers[0]),  # Batch normalization
        )

        # Final output layer (with sigmoid activation for binary segmentation)
        self.aux = nn.Sequential(
            nn.Conv2d(layers[0], n_classes, kernel_size=1),  # Convolution to output final predictions
            nn.BatchNorm2d(n_classes),  # Batch normalization
            nn.Sigmoid()  # Sigmoid activation for output probabilities
        )
        h, w = input_resolution

        # Attention modules (DGM), applied at various stages of the network
        self.att = DGM(32, (h//2,w//2))  # Attention applied after first down block
        self.att2 = DGM(32, (h//4,w//4))  # Attention applied after second down block
        self.att3 = DGM(32, (h//4,w//4))  # Attention applied after first up block
        self.att4 = DGM(32, (h//2,w//2))  # Attention applied after second up block

    def forward(self, x):
        # Apply the first convolutional layer
        x_first = self.first(x)

        # Downsampling path with attention applied after each step
        x1 = self.down_path[0](x_first)  # First downsampling block
        x1 = self.att(x1)  # Apply attention
        x2 = self.down_path[1](x1)  # Second downsampling block
        x2 = self.att2(x2)  # Apply attention
        x3 = self.down_path[2](x2)  # Third downsampling block
        x4 = self.down_path[3](x3)  # Fourth downsampling block

        # Upsampling path with attention applied after each step
        d1 = self.up_path[0](x4, x3)  # First upsampling block, with skip connection from x3
        d2 = self.up_path[1](d1, x2)  # Second upsampling block, with skip connection from x2
        d2 = self.att3(d2)  # Apply attention
        d3 = self.up_path[2](d2, x1)  # Third upsampling block, with skip connection from x1
        d3 = self.att4(d3)  # Apply attention
        d4 = self.up_path[3](d3, x_first)  # Final upsampling block, with skip connection from x_first

        # Final convolution and batch normalization
        x = self.conv_bn(d4)

        # Save features for potential future use (like visualization)
        self.feat = x

        # Apply the final auxiliary network to get the segmentation prediction
        self.pred = self.aux(x)
        return self.pred

# Example usage:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DGNet(inp_c=3,input_resolution=(128,128)).to(device)
    input_tensor = torch.randn(2, 3, 128, 128).to(device)
    output = model(input_tensor)
    print("Output shape:", output.shape)
