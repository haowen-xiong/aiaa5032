DEFAULT_BLOCK_SPECS = ((1, 32, 64), (64, 32, 128))
DEFAULT_GRAPH_CONV_TYPE = "cheb"
DEFAULT_DIRECT_MULTI_STEP = False


def normalize_block_specs(block_specs=None):
    specs = DEFAULT_BLOCK_SPECS if block_specs is None else tuple(tuple(int(v) for v in spec) for spec in block_specs)
    if not specs:
        raise ValueError("block_specs must contain at least one block")
    for spec in specs:
        if len(spec) != 3:
            raise ValueError("each block spec must contain exactly three integers: [c_in, c_t, c_out]")
        if any(v <= 0 for v in spec):
            raise ValueError("block spec values must be positive")
    return specs
