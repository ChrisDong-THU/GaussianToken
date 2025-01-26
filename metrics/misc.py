import numpy as np
from scipy import linalg
from PIL import Image


def normalize(x):
    x = (x + 1) / 2
    
    return x.clamp(0, 1)

def custom_to_pil(x):
    x = x.detach().cpu()
    x = x.permute(1, 2, 0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

