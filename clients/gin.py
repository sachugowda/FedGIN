# common/gin.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ====================== GIN (https://github.com/cheng-01037/Causality-Medical-Image-Domain-Generalization/blob/main/models/imagefilter.py) ====================== #


class GradlessGCReplayNonlinBlock(nn.Module):
    def __init__(self, out_channel=32, in_channel=1, scale_pool=[1, 3], layer_id=0, use_act=True, requires_grad=False, device='cpu', **kwargs):
        """
        Conv-leaky ReLU layer with group convolutions.
        - `in_channel=1` since your dataset is grayscale (MRI/CT images)
        """
        super(GradlessGCReplayNonlinBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.scale_pool = scale_pool
        self.layer_id = layer_id
        self.use_act = use_act
        self.requires_grad = requires_grad
        self.device = device
        assert requires_grad is False  # Ensure no gradients

    def forward(self, x_in):
        """
        Args:
            x_in: [batch_size (nb), channels (nc=1), height (nx=256), width (ny=256)]
        """
        # Ensure input is on the correct device and dtype
        x_in = x_in.to(self.device, dtype=x_in.dtype)
        nb, nc, nx, ny = x_in.shape

        # Ensure grayscale images stay 1-channel
        if nc != self.in_channel:
            raise ValueError(f"Expected {self.in_channel} channels, but got {nc}")

        # Random kernel selection
        idx_k = torch.randint(high=len(self.scale_pool), size=(1,), dtype=torch.long, device=self.device)
        k = self.scale_pool[idx_k.item()]

        dtype = x_in.dtype  # Use input dtype
        ker = torch.randn([self.out_channel, self.in_channel, k, k], requires_grad=self.requires_grad, device=self.device, dtype=dtype)
        shift = torch.randn([self.out_channel, 1, 1], requires_grad=self.requires_grad, device=self.device, dtype=dtype) * 1.0

        # Apply convolution
        x_conv = F.conv2d(x_in, ker, stride=1, padding=k // 2, dilation=1, groups=1)
        x_conv = x_conv + shift

        if self.use_act:
            x_conv = F.leaky_relu(x_conv)

        x_conv = x_conv.view(nb, self.out_channel, nx, ny)
        return x_conv


class GINGroupConv(nn.Module):
    def __init__(self, out_channel=1, in_channel=1, interm_channel=2, scale_pool=[1, 3], n_layer=4, 
                 out_norm='frob', use_custom_block=True, device='cpu'):
        """
        GIN with configurable convolutional layers.
        - `in_channel=1` for grayscale MRI/CT images
        - `use_custom_block`: If True, uses `GradlessGCReplayNonlinBlock`; otherwise, `nn.Conv2d`.
        - `device`: Specifies the device to move the model to.
        """
        super(GINGroupConv, self).__init__()
        self.scale_pool = scale_pool  
        self.n_layer = n_layer
        self.out_norm = out_norm
        self.out_channel = out_channel
        self.device = device
        self.use_custom_block = use_custom_block

        # Choose block type
        self.block_type = GradlessGCReplayNonlinBlock if use_custom_block else self._conv_block
        
        self.layers = nn.ModuleList()
        self.layers.append(
            self.block_type(out_channel=interm_channel, in_channel=in_channel, scale_pool=scale_pool, layer_id=0, device=device).to(device)
        )
        for ii in range(n_layer - 2):
            self.layers.append(
                self.block_type(out_channel=interm_channel, in_channel=interm_channel, scale_pool=scale_pool, layer_id=ii + 1, device=device).to(device)
            )
        self.layers.append(
            self.block_type(out_channel=out_channel, in_channel=interm_channel, scale_pool=scale_pool, layer_id=n_layer - 1, use_act=False, device=device).to(device)
        )

    def _conv_block(self, out_channel, in_channel, scale_pool, layer_id, use_act=True):
        """ Standard Conv2D Block (Alternative to GradlessGCReplayNonlinBlock) """
        kernel_size = scale_pool[layer_id % len(scale_pool)]  # Choose from scale_pool
        padding = kernel_size // 2
        conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)
        if use_act:
            return nn.Sequential(conv, nn.BatchNorm2d(out_channel), nn.LeakyReLU(0.2))
        else:
            return conv  # No activation in the final layer

    def forward(self, x_in):
        x_in = x_in.to(self.device)  # Ensure input tensor is on correct device

        if isinstance(x_in, list):
            x_in = torch.cat(x_in, dim=0)

        nb, nc, nx, ny = x_in.shape
        alphas = torch.rand(nb, device=self.device).view(-1, 1, 1, 1)  # Shape: (nb, 1, 1, 1)

        x = self.layers[0](x_in)
        for blk in self.layers[1:]:
            x = blk(x)

        # Ensure `x` is also on the correct device before interpolation
        x = x.to(self.device)

        # Optimized alpha interpolation using torch.lerp()
        mixed = torch.lerp(x_in, x, alphas)

        # Frobenius norm-based normalization
        if self.out_norm == 'frob':
            _in_frob = torch.norm(x_in.view(nb, nc, -1), dim=(-1, -2), p='fro', keepdim=False).view(nb, 1, 1, 1).to(self.device)
            _self_frob = torch.norm(mixed.view(nb, self.out_channel, -1), dim=(-1, -2), p='fro', keepdim=False).view(nb, 1, 1, 1).to(self.device)
            mixed = mixed * (1.0 / (_self_frob + 1e-5)) * _in_frob

        return mixed


# ====================== GIN Wrapper Function ====================== #
def apply_gin(image_tensor, out_channel=1, in_channel=1, interm_channel=2, scale_pool=[1, 3], n_layer=4, 
              out_norm='frob', use_custom_block=True, device='cpu'):
    """ Apply the optimized GIN augmentation for grayscale images (1-channel). """
    gin = GINGroupConv(out_channel, in_channel, interm_channel, scale_pool, n_layer, 
                       out_norm, use_custom_block, device).to(device)
    return gin(image_tensor.to(device))
