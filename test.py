import torch

from accelerate import Accelerator
a=1
print(torch.__version__)
print(torch.cuda.is_available())

print(f"device: {Accelerator().device}")