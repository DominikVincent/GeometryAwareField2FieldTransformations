import gc

import torch

from nerfstudio.model_components.nesf_components import *

models = [TranformerEncoderModelConfig(
    num_layers=8,
    num_heads=8,
    dim_feed_forward=128,
    feature_dim=128,
    ).setup(input_size=103) for _ in range(10)]
# models = [StratifiedTransformerWrapperConfig().setup(input_size=103) for _ in range(10)]
model = models[0]


# model prameters
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

S = 2048

for i in range(10):
    model = models[i]
    model.train()
    model.to("cuda:0")
    input_data = torch.randn(1, S, 103, device="cuda:0", requires_grad=True)
    points_xyz = torch.randn(1, S , 3, device="cuda:0", requires_grad=False)
    output_data = torch.randn(1, S, 48, device="cuda:0")
    out = model(input_data, batch={"points_xyz": points_xyz})
    print(f"Current memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Current memory cached: {torch.cuda.memory_cached() / 1024**2:.2f} MB")
    print(f"Current memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"Current memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    gc.collect()
    torch.cuda.empty_cache()
    print(f" After clear  - Current memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f" After clear  - Current memory cached: {torch.cuda.memory_cached() / 1024**2:.2f} MB")
    print(f" After clear  - Current memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f" After clear  - Current memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    model.to("cpu")


print(torch.cuda.memory_summary())

input("Press Enter to continue...")
print(torch.cuda.memory_summary())
