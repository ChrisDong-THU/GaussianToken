import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def draw_tensors(tensors, max_cols=4):
    imgs = tensors.detach().permute(0, 2, 3, 1).cpu().numpy()
    batch_size = imgs.shape[0]
    n_cols = min(batch_size, max_cols)

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    for i in range(n_cols):
        img = imgs[i]
        axes[i].imshow(img)
        axes[i].axis('off')

    plt.show()
    

def draw_gaussians_params_1d(params, value_range=[0, 1]):
    plt.figure()
    plot_x = np.arange(params.shape[0])
    params = np.sort(params)
    plt.plot(plot_x, params)
    
    plt.ylim(value_range[0], value_range[1])

    return plt.gcf()


def draw_gaussians_params_2d(params, idx=None, value_range=[0, 1]):
    plt.figure()

    plt.scatter(params[:, 0], params[:, 1], color='grey', marker='o')
    
    if not idx is None:
        for i in range(params.shape[0]):
            plt.text(params[i, 0], params[i, 1], str(idx[i]), fontsize=8, color='black', ha='center', va='center')

    # 设置坐标轴范围
    plt.xlim(value_range[0], value_range[1])
    plt.ylim(value_range[0], value_range[1])

    return plt.gcf()


def draw_codebook_usage(code_counter, codebook_size):
    plt.figure()
    plot_x = np.arange(codebook_size)
    freq = torch.bincount(torch.cat(code_counter, dim=0).view(-1), minlength=codebook_size)
    freq, _ = torch.sort(freq/freq.sum())
    freq_integration = np.cumsum(freq.cpu().numpy())
    percent_index = np.argmin(np.abs(freq_integration - 0.2))
    percent_freq_x = plot_x[percent_index]
    plt.plot(plot_x, freq.cpu().numpy(), label='Frequency')
    plt.axvline(x=percent_freq_x, color='r', linestyle='--', label='Percent Line')
    
    return plt.gcf()


def save_normlized_tensor(tensor, path):
    '''
    Save tensor in range [0, 1] as image
    '''
    x = tensor.detach().cpu()
    x = x.permute(1, 2, 0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    x.save(path)