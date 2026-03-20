import torch
import torch.nn as nn
import torch.nn.functional as F


def align_channels(x, c_out, projection=None):
    c_in = x.size(-1)
    if c_in > c_out:
        if projection is None:
            raise ValueError("projection is required when c_in > c_out")
        return projection(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    if c_in < c_out:
        pad = x.new_zeros(x.size(0), x.size(1), x.size(2), c_out - c_in)
        return torch.cat([x, pad], dim=-1)
    return x


class LayerNorm2D(nn.Module):
    def __init__(self, n_route, channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, 1, n_route, channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, n_route, channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = ((x - mean) ** 2).mean(dim=(2, 3), keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta


class TemporalConvLayer(nn.Module):
    def __init__(self, Kt, c_in, c_out, act_func="relu"):
        super().__init__()
        self.Kt = Kt
        self.c_out = c_out
        self.act_func = act_func
        self.input_projection = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=False) if c_in > c_out else None
        out_channels = 2 * c_out if act_func == "GLU" else c_out
        self.conv = nn.Conv2d(c_in, out_channels, kernel_size=(Kt, 1), bias=True)

    def forward(self, x):
        _, T, _, _ = x.shape
        x_input = align_channels(x, self.c_out, self.input_projection)
        x_input = x_input[:, self.Kt - 1:T, :, :]
        x_conv = self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        if self.act_func == "GLU":
            lhs, rhs = torch.split(x_conv, self.c_out, dim=-1)
            return (lhs + x_input) * torch.sigmoid(rhs)
        if self.act_func == "linear":
            return x_conv
        if self.act_func == "sigmoid":
            return torch.sigmoid(x_conv)
        if self.act_func == "relu":
            return F.relu(x_conv + x_input)
        raise ValueError(f'ERROR: activation function "{self.act_func}" is not defined.')


class ChebSpatialConvLayer(nn.Module):
    def __init__(self, Ks, c_in, c_out, graph_kernel):
        super().__init__()
        kernel = torch.as_tensor(graph_kernel, dtype=torch.float32)
        if kernel.ndim != 2:
            raise ValueError("cheb graph_kernel must be a 2D tensor")
        self.Ks = Ks
        self.c_in = c_in
        self.c_out = c_out
        self.register_buffer("graph_kernel", kernel)
        self.theta = nn.Parameter(torch.empty(Ks * c_in, c_out))
        self.bias = nn.Parameter(torch.zeros(c_out))
        nn.init.xavier_uniform_(self.theta)
        self.input_projection = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=False) if c_in > c_out else None

    def forward(self, x):
        batch_size, T, n, _ = x.shape
        x_input = align_channels(x, self.c_out, self.input_projection)
        x_bt = x.reshape(-1, n, self.c_in)
        x_tmp = x_bt.transpose(1, 2).contiguous().reshape(-1, n)
        x_mul = torch.matmul(x_tmp, self.graph_kernel).reshape(-1, self.c_in, self.Ks, n)
        x_ker = x_mul.permute(0, 3, 1, 2).reshape(-1, self.c_in * self.Ks)
        x_gconv = torch.matmul(x_ker, self.theta).reshape(batch_size, T, n, self.c_out)
        x_gc = x_gconv + self.bias.view(1, 1, 1, -1)
        return F.relu(x_gc + x_input)


class FirstOrderSpatialConvLayer(nn.Module):
    def __init__(self, c_in, c_out, graph_kernel):
        super().__init__()
        kernel = torch.as_tensor(graph_kernel, dtype=torch.float32)
        if kernel.ndim != 2:
            raise ValueError("first-order graph_kernel must be a 2D tensor")
        self.c_in = c_in
        self.c_out = c_out
        self.register_buffer("graph_kernel", kernel)
        self.theta = nn.Parameter(torch.empty(c_in, c_out))
        self.bias = nn.Parameter(torch.zeros(c_out))
        nn.init.xavier_uniform_(self.theta)
        self.input_projection = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=False) if c_in > c_out else None

    def forward(self, x):
        batch_size, T, n, _ = x.shape
        x_input = align_channels(x, self.c_out, self.input_projection)
        x_bt = x.reshape(-1, n, self.c_in)
        x_agg = torch.einsum("mn,bnc->bmc", self.graph_kernel, x_bt)
        x_gconv = torch.matmul(x_agg.reshape(-1, self.c_in), self.theta).reshape(batch_size, T, n, self.c_out)
        x_gc = x_gconv + self.bias.view(1, 1, 1, -1)
        return F.relu(x_gc + x_input)


class IdentitySpatialLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.c_out = c_out
        self.input_projection = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=False) if c_in > c_out else None

    def forward(self, x):
        return align_channels(x, self.c_out, self.input_projection)


class STConvBlock(nn.Module):
    def __init__(self, Ks, Kt, channels, n_route, graph_kernel, drop_prob=0.0, act_func="GLU", graph_conv_type="cheb", use_spatial=True):
        super().__init__()
        c_si, c_t, c_oo = channels
        self.temp1 = TemporalConvLayer(Kt, c_si, c_t, act_func=act_func)
        if use_spatial:
            if graph_conv_type == "cheb":
                self.spat = ChebSpatialConvLayer(Ks, c_t, c_t, graph_kernel)
            elif graph_conv_type == "first":
                self.spat = FirstOrderSpatialConvLayer(c_t, c_t, graph_kernel)
            else:
                raise ValueError(f'Unsupported graph_conv_type "{graph_conv_type}"')
        else:
            self.spat = IdentitySpatialLayer(c_t, c_t)
        self.temp2 = TemporalConvLayer(Kt, c_t, c_oo)
        self.norm = LayerNorm2D(n_route, c_oo)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.temp1(x)
        x = self.spat(x)
        x = self.temp2(x)
        x = self.norm(x)
        return self.dropout(x)


class FullyConvLayer(nn.Module):
    def __init__(self, n_route, channel):
        super().__init__()
        self.conv = nn.Conv2d(channel, 1, kernel_size=(1, 1), bias=False)
        self.bias = nn.Parameter(torch.zeros(1, 1, n_route, 1))

    def forward(self, x):
        out = self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return out + self.bias


class OutputLayer(nn.Module):
    def __init__(self, T, n_route, channel, act_func="GLU"):
        super().__init__()
        self.temp1 = TemporalConvLayer(T, channel, channel, act_func=act_func)
        self.norm = LayerNorm2D(n_route, channel)
        self.temp2 = TemporalConvLayer(1, channel, channel, act_func="sigmoid")
        self.fc = FullyConvLayer(n_route, channel)

    def forward(self, x):
        x = self.temp1(x)
        x = self.norm(x)
        x = self.temp2(x)
        return self.fc(x)


class DirectMultiStepOutput(nn.Module):
    def __init__(self, T, n_route, channel, n_pred, act_func="GLU"):
        super().__init__()
        self.n_pred = n_pred
        self.temp1 = TemporalConvLayer(T, channel, channel, act_func=act_func)
        self.norm = LayerNorm2D(n_route, channel)
        self.temp2 = TemporalConvLayer(1, channel, channel, act_func="sigmoid")
        self.head = nn.Conv2d(channel, n_pred, kernel_size=(1, 1), bias=True)

    def forward(self, x):
        x = self.temp1(x)
        x = self.norm(x)
        x = self.temp2(x)
        x = self.head(x.permute(0, 3, 1, 2)).permute(0, 1, 3, 2)
        return x
