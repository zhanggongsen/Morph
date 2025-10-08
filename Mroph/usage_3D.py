import torch
from networks import SwinTransformer3D, SwinTransformerConfig3D

config = SwinTransformerConfig3D(
    input_size=(64, 160, 224),
    in_channels=1,
    embed_dim=16,
    num_blocks=[2, 2, 2,2],
    patch_window_size=[(1, 1, 1), (2, 2, 2), (2, 2, 2),(2, 2, 2)],
    block_window_size=[(2, 7, 7), (2, 7, 7), (2, 7, 7),(2, 7, 7)],
    num_heads=[2, 2, 2,2],
)


model = SwinTransformer3D(config)
x = torch.randn(1, 1,64, 160, 224)
output = model(x)
for i in range(len(output)):
    print(output[i].shape)