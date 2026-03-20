import torch
import torch.nn as nn

from .config import DEFAULT_BLOCK_SPECS, DEFAULT_DIRECT_MULTI_STEP, DEFAULT_GRAPH_CONV_TYPE, normalize_block_specs
from .layers import DirectMultiStepOutput, OutputLayer, STConvBlock


class STGCN(nn.Module):
    def __init__(
        self,
        n_his,
        Ks,
        Kt,
        n_route,
        graph_kernel,
        block_specs=None,
        drop_prob=0.0,
        graph_conv_type=DEFAULT_GRAPH_CONV_TYPE,
        use_spatial=True,
        direct_multi_step=DEFAULT_DIRECT_MULTI_STEP,
        n_pred=None,
        output_act_func="GLU",
    ):
        super().__init__()
        self.n_his = int(n_his)
        self.n_route = int(n_route)
        self.graph_conv_type = graph_conv_type
        self.use_spatial = bool(use_spatial)
        self.direct_multi_step = bool(direct_multi_step)
        self.n_pred = n_pred

        specs = normalize_block_specs(block_specs if block_specs is not None else DEFAULT_BLOCK_SPECS)
        self.block_specs = specs
        self.blocks = nn.ModuleList(
            [
                STConvBlock(
                    Ks=Ks,
                    Kt=Kt,
                    channels=channels,
                    n_route=self.n_route,
                    graph_kernel=graph_kernel,
                    drop_prob=drop_prob,
                    act_func="GLU",
                    graph_conv_type=self.graph_conv_type,
                    use_spatial=self.use_spatial,
                )
                for channels in specs
            ]
        )

        ko = self.n_his
        for _ in specs:
            ko -= 2 * (Kt - 1)
        if ko <= 1:
            raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{ko}".')

        last_channels = specs[-1][-1]
        if self.direct_multi_step:
            if self.n_pred is None:
                raise ValueError("n_pred must be provided when direct_multi_step=True")
            self.output = DirectMultiStepOutput(ko, self.n_route, last_channels, int(self.n_pred), act_func=output_act_func)
        else:
            self.output = OutputLayer(ko, self.n_route, last_channels, act_func=output_act_func)

    def forward(self, inputs):
        x = inputs[:, 0 : self.n_his, :, :]
        for block in self.blocks:
            x = block(x)
        y = self.output(x)
        if self.direct_multi_step:
            return y
        return y[:, 0, :, :]


def build_stgcn(
    n_his,
    Ks,
    Kt,
    n_route,
    graph_kernel,
    block_specs=None,
    drop_prob=0.0,
    graph_conv_type=DEFAULT_GRAPH_CONV_TYPE,
    use_spatial=True,
    direct_multi_step=DEFAULT_DIRECT_MULTI_STEP,
    n_pred=None,
    output_act_func="GLU",
):
    return STGCN(
        n_his=n_his,
        Ks=Ks,
        Kt=Kt,
        n_route=n_route,
        graph_kernel=graph_kernel,
        block_specs=block_specs,
        drop_prob=drop_prob,
        graph_conv_type=graph_conv_type,
        use_spatial=use_spatial,
        direct_multi_step=direct_multi_step,
        n_pred=n_pred,
        output_act_func=output_act_func,
    )
