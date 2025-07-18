from .base import Layout


class GFX950MXScaleLayout(Layout):
    name: str = "GFX950_SCALE"

    def __init__(self, shape) -> None:
        super().__init__(shape)

    def swizzle_data(self, data):
        data = data.transpose(-1, -2).contiguous()
        E, M, SCALE_K = data.shape
        data = data.view(E, M // 32, 2, 16, SCALE_K // 8, 2, 4, 1)
        data = data.permute(0, 1, 4, 6, 3, 5, 2, 7).contiguous()
        data = data.reshape(E, M // 32, SCALE_K * 32)
        # return data.transpose(-1, -2).contiguous()
        return data.transpose(-1, -2)

    def unswizzle_data(self, data):
        raise NotImplementedError()

    def swizzle_block_shape(self, block_shape):
        E, SCALE_K, M = block_shape
        return [E, M // 32, SCALE_K * 32]
