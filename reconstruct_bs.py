import lightning as L
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import argparse
from omegaconf import OmegaConf
from tqdm import tqdm

from gstk.models.gqgan import GQGAN
from gstk.data.dataset import MiniImagenet
from gstk.data.imagenet import ImageNetValidation


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
    model = load_gqgan(config, ckpt_path=args.ckpt_path)

    # dataset = MiniImagenet(split='val')
    dataset = ImageNetValidation(config={'size': 256, 'subset': None})
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            images = batch["image"].cuda()
            
            if model.use_ema:
                with model.ema_scope():
                    reconstructed_images, indices, loss_commit, gaussians = model(images)
            else:
                reconstructed_images, indices, loss_commit, gaussians = model(images)
                
            save_image(reconstructed_images, f"./tmp_rec/{i}.png", nrow=8, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    args = get_args()
    main(args)