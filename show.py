import torch
from eval import val_

model = torch.load("best_0.9113741187023321.pth")
val_(model)
