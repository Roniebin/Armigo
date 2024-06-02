import torch

print(torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
#  Returns a bool indicating if CUDA is currently available.
print(torch.cuda.is_available())
