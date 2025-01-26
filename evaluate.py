"""
We provide Tokenizer Evaluation code here.
Refer to 
https://github.com/richzhang/PerceptualSimilarity
https://github.com/mseitzer/pytorch-fid
"""

import lightning as L
import torch
from torchvision.utils import save_image

import argparse
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

from gstk.models.gqgan import GQGAN
from gstk.data.from_config import instantiate_from_config
from gstk.modules.gaussianembed.draw import draw_gaussians_params_2d

import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from metrics.inception import InceptionV3, calculate_frechet_distance
from metrics.misc import normalize


torch.cuda.set_device("cuda:0")
L.seed_everything(0)
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.deterministic = True #True
torch.backends.cudnn.benchmark = False #False


def load_gqgan(config, ckpt_path=None):
    model = GQGAN(**config.model.init_args)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(sd, strict=False)
    
    model = model.cuda()

    return model.eval()

def get_args():
    parser = argparse.ArgumentParser(description="inference parameters")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--batch_size", default=64, type=int)

    return parser.parse_args()

def main(args):
    config = OmegaConf.load(args.config_file)
    config.data.init_args.batch_size = args.batch_size

    dataset = instantiate_from_config(config.data)
    dataset.prepare_data()
    dataset.setup()
    pred_xs = []
    pred_recs = []
    
    model = load_gqgan(config, ckpt_path=args.ckpt_path)
    codebook_size = config.model.init_args.vq_cfg.codebook_size
    
    #usage
    usage = np.zeros(codebook_size)

    # FID score related
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx]).cuda()
    inception_model.eval()

    # LPIPS score related
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()  # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()   # closer to "traditional" perceptual loss, when used for optimization
    lpips_alex = 0.0
    lpips_vgg = 0.0

    ssim_value = 0.0
    psnr_value = 0.0

    num_images = 0
    num_iter = 0
    with torch.no_grad():
        for batch in tqdm(dataset._val_dataloader()):
            images = batch["image"].cuda()
            
            save_image(images, f"./tmp/ori.png", nrow=4, normalize=True, value_range=(-1, 1))
            
            num_images += images.shape[0]

            if model.use_ema:
                with model.ema_scope():
                    reconstructed_images, indices, loss_commit, gaussians = model(images)
            else:
                reconstructed_images, indices, loss_commit, gaussians = model(images)
            
            save_image(reconstructed_images, f"./tmp/rec.png", nrow=4, normalize=True, value_range=(-1, 1))
            
            indices = indices.cpu().numpy()
            code_idx = indices[0]
            means = gaussians.means[0].cpu().numpy()
            
            gaussian_fig = draw_gaussians_params_2d(means, code_idx, value_range=[-1, 1])
            gaussian_fig.savefig('./tmp/gaussian.png')
            plt.close(gaussian_fig)
            
            ### usage
            for index in indices:
                usage[index] += 1
            
            # calculate lpips
            lpips_alex += loss_fn_alex(images, reconstructed_images).sum()
            lpips_vgg += loss_fn_vgg(images, reconstructed_images).sum()

            images = normalize(images)
            reconstructed_images = normalize(reconstructed_images)

            # calculate fid
            pred_x = inception_model(images)[0]
            pred_x = pred_x.squeeze(3).squeeze(2).cpu().numpy()
            pred_rec = inception_model(reconstructed_images)[0]
            pred_rec = pred_rec.squeeze(3).squeeze(2).cpu().numpy()

            pred_xs.append(pred_x)
            pred_recs.append(pred_rec)

            #calculate PSNR and SSIM
            rgb_restored = (reconstructed_images * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            rgb_gt = (images * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            rgb_restored = rgb_restored.astype(np.float32) / 255.
            rgb_gt = rgb_gt.astype(np.float32) / 255.
            ssim_temp = 0
            psnr_temp = 0
            B, _, _, _ = rgb_restored.shape
            for i in range(B):
                rgb_restored_s, rgb_gt_s = rgb_restored[i], rgb_gt[i]
                ssim_temp += ssim_loss(rgb_restored_s, rgb_gt_s, data_range=1.0, channel_axis=-1)
                psnr_temp += psnr_loss(rgb_gt, rgb_restored)
            ssim_value += ssim_temp / B
            psnr_value += psnr_temp / B
            num_iter += 1

    pred_xs = np.concatenate(pred_xs, axis=0)
    pred_recs = np.concatenate(pred_recs, axis=0)

    mu_x = np.mean(pred_xs, axis=0)
    sigma_x = np.cov(pred_xs, rowvar=False)
    mu_rec = np.mean(pred_recs, axis=0)
    sigma_rec = np.cov(pred_recs, rowvar=False)
    fid_value = calculate_frechet_distance(mu_x, sigma_x, mu_rec, sigma_rec)
    
    lpips_alex_value = lpips_alex / num_images
    lpips_vgg_value = lpips_vgg / num_images
    
    ssim_value = ssim_value / num_iter
    psnr_value = psnr_value / num_iter

    num_count = np.count_nonzero(usage)
    utilization = num_count / codebook_size

    print("FID: ", fid_value)
    print("LPIPS_ALEX: ", lpips_alex_value.item())
    print("LPIPS_VGG: ", lpips_vgg_value.item())
    print("SSIM: ", ssim_value)
    print("PSNR: ", psnr_value)
    print(f"Codebook: {utilization * 100:.2f}%")


if __name__ == "__main__":
    args = get_args()
    main(args)