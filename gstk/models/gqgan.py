import lightning as L
from timm.optim import create_optimizer_v2 as create_optimizer
from timm.scheduler import create_scheduler_v2 as create_scheduler
from contextlib import contextmanager

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from ..modules.diffusionmodules.improved_model import Encoder, Decoder
# from ..modules.gaussianembed.vq import VectorQuantize
from ..modules.gaussianembed.vq_v2 import VectorQuantize
from ..modules.gaussianembed.gaussian_embed import GaussianEmbed
from ..modules.gaussianembed.gaussian_render import RenderSet, render_gaussians
from ..modules.ema import LitEma

from ..modules.losses.vqperceptual import VQLPIPSWithDiscriminator
from ..modules.gaussianembed.draw import draw_gaussians_params_1d, draw_gaussians_params_2d, draw_codebook_usage, save_normlized_tensor

from metrics.misc import normalize
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio


class GQGAN(L.LightningModule):
    def __init__(self, fm_shape=None, img_encoder_cfg=None, img_decoder_cfg=None, z_channels=None, vq_cfg=None, gs_embed_cfg=None, 
                 image_key="image", use_ema=False, loss_cfg=None, optim_gen_cfg=None, optim_disc_cfg=None, lr_scheduler_cfg=None, 
                 embed_dim=256, num_gs=256, xy_range=[-1, 1], initial_scale_range=[0.1, 0.5], final_scale_range=[0.1, 0.5], include_opa=False):
        super().__init__()
        
        self.image_key = image_key
        self.img_encoder = Encoder(**img_encoder_cfg)
        self.img_decoder = Decoder(**img_decoder_cfg)
        self.quantize = VectorQuantize(**vq_cfg)
        
        self.gs_embed = GaussianEmbed(**gs_embed_cfg) if gs_embed_cfg else None
        self.render_set = RenderSet(img_H=fm_shape[0], img_W=fm_shape[1], block_H=16, block_W=16).post_set(z_channels)
        
        self.loss = VQLPIPSWithDiscriminator(**loss_cfg)
        
        self.use_ema = use_ema
        if use_ema:
            self.model_ema = LitEma(self)
        
        self.optim_gen_cfg = optim_gen_cfg
        self.optim_disc_cfg = optim_disc_cfg
        self.lr_scheduler_cfg = lr_scheduler_cfg
        
        self.automatic_optimization = False # 多个优化器时，需要手动优化
        self.strict_loading = False
        
        self.fixed_input = None
        self.code_counter = []
        
        # evaluate metrics
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).eval()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).eval()
        self.fid = FrechetInceptionDistance(feature=2048, input_img_size=(3, 256, 256), normalize=True).eval()
        self.inception = InceptionScore(feature=2048, normalize=True).eval()
        self.lpip = LearnedPerceptualImagePatchSimilarity(normalize=True).eval()
    
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        '''
        Filter out non-training components such as evaluation metrics
        '''
        # Get the full state dict
        state_dict = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        
        # Remove the evaluation metrics from the state dict
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                               if not any(metric_name in k for metric_name in ["ssim", "psnr", "fid", "inception", "lpip"])}
        
        return filtered_state_dict
    
    def get_last_layer(self):
        return self.img_decoder.conv_out.weight
    
    def encode(self, x):
        fm_raw = self.img_encoder(x)
        trainer = self.trainer if self.training else None
        
        if self.gs_embed:
            gaussian = self.gs_embed(fm_raw)
            feature_quant, indices, loss_commit = self.quantize(trainer, gaussian.features)
            # feature_quant, indices, loss_commit = gaussian.features, torch.tensor(0), torch.tensor(0.0)
            gaussian.features = feature_quant
            fm_quant = render_gaussians(gaussian, self.render_set)
        else:
            gaussian = None
            fm_quant, indices, loss_commit = self.quantize(trainer, fm_raw)
            
        return fm_quant, indices, loss_commit, gaussian
    
    def post_process(self, gaussian):
        out = render_gaussians(gaussian, self.render_set)
        
        return out
    
    def decode(self, fm_quant):
        dec = self.img_decoder(fm_quant)
        
        return dec
    
    def decode_gaussian(self, gaussian):
        fm_quant = render_gaussians(gaussian, self.render_set)
        dec = self.img_decoder(fm_quant)
        
        return dec
    
    def decode_code(self, indices):
        fm_quant = self.quantize.indices_to_codes(indices)
        dec = self.img_decoder(fm_quant)
        
        return dec
    
    def forward(self, x):
        fm_quant, indices, loss_commit, gaussian = self.encode(x)
        x_tilde = self.decode(fm_quant)
        
        return x_tilde, indices, loss_commit, gaussian
    
    def get_input(self, batch, key):
        x = batch[key]
        if x.dim() == 3:
            x = x.unsqueeze(1) # add channel dim

        return x.float()
    
    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, indices, loss_commit, gaussians = self(x)

        opt_gen, opt_disc = self.optimizers()
        scheduler_gen, scheduler_disc = self.lr_schedulers()

        # fix global step bug, ref. https://github.com/Lightning-AI/pytorch-lightning/issues/17958
        opt_disc._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        opt_disc._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        
        # optimize generator
        aeloss, log_dict_ae = self.loss(loss_commit, x, xrec, 0, self.current_epoch, last_layer=self.get_last_layer(), split="train")
        opt_gen.zero_grad()
        self.manual_backward(aeloss)
        opt_gen.step()
        
        scheduler_gen.step_update(self.global_step)
        
        # optimize discriminator
        discloss, log_dict_disc = self.loss(loss_commit, x, xrec, 1, self.current_epoch, last_layer=self.get_last_layer(), split="train")
        opt_disc.zero_grad()
        self.manual_backward(discloss)
        opt_disc.step()
        
        scheduler_disc.step_update(self.global_step)

        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)
    
    def on_train_epoch_end(self):
        scheduler_gen, scheduler_disc = self.lr_schedulers()
        
        scheduler_gen.step(self.current_epoch)
        scheduler_disc.step(self.current_epoch)
    
    def validation_step(self, batch, batch_idx): 
        if self.fixed_input is None: # first step
            self.fixed_input = self.get_input(batch, self.image_key) # for logging
            fixed_grid = make_grid(self.fixed_input, nrow=8, value_range=(-1, 1), normalize=True)
            self.logger.experiment.add_image("image/ori", fixed_grid, 0)
        
        if self.use_ema:
            with self.ema_scope():
                self._validation_step(batch, suffix="_ema")
        else:
            self._validation_step(batch, suffix="_no_ema")

    def _validation_step(self, batch, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, indices, loss_commit, gaussians = self(x)
        self.code_counter.append(indices)
        
        aeloss, log_dict_ae = self.loss(loss_commit, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val"+ suffix)

        discloss, log_dict_disc = self.loss(loss_commit, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val" + suffix)
        
        x, xrec = normalize(x), normalize(xrec)
        
        # ssim and psnr
        self.ssim.update(xrec, x)
        self.psnr.update(xrec, x)
        
        # fid and inception
        self.fid.update(x, real=True)
        self.fid.update(xrec, real=False)
        self.inception.update(xrec)
        self.lpip.update(xrec, x)
    
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        
    def on_validation_epoch_end(self):
        codebook_usage_fig = draw_codebook_usage(self.code_counter, self.quantize.codebook_size)
        self.logger.experiment.add_figure("quantize/codebook_usage", codebook_usage_fig, self.current_epoch)
        plt.close(codebook_usage_fig)
        self.code_counter = []
        
        fixed_xrec, indices, _, fixed_gaussians = self(self.fixed_input)
        fixed_grid = make_grid(fixed_xrec.cpu(), nrow=8, value_range=(-1, 1), normalize=True)
        self.logger.experiment.add_image("image/rec", fixed_grid, self.current_epoch)
        
        if fixed_gaussians:
            means = fixed_gaussians.means[0].detach().cpu().numpy()
            indices = indices[0].detach().cpu().numpy()
            means_fig = draw_gaussians_params_2d(means, indices, value_range=[-1, 1])
            self.logger.experiment.add_figure("gaussian/means", means_fig, self.current_epoch)
            plt.close(means_fig)
            scales = fixed_gaussians.scales[0].detach().cpu().numpy()
            scales_fig = draw_gaussians_params_2d(scales, value_range=[0.1, 4])
            self.logger.experiment.add_figure("gaussian/scales", scales_fig, self.current_epoch)
            plt.close(scales_fig)
        
        # ssim and psnr
        ssim_score = self.ssim.compute()
        psnr_score = self.psnr.compute()
        self.ssim.reset()
        self.psnr.reset()
        
        self.logger.experiment.add_scalar("metrics/ssim", ssim_score, self.current_epoch)
        self.logger.experiment.add_scalar("metrics/psnr", psnr_score, self.current_epoch)
        
        # fid and inception
        fid_score = self.fid.compute()
        inception_score = self.inception.compute()[0] # only mean value
        lpips_score = self.lpip.compute()
        self.fid.reset()
        self.inception.reset()
        self.lpip.reset()
        
        self.logger.experiment.add_scalar("metrics/fid", fid_score, self.current_epoch)
        self.logger.experiment.add_scalar("metrics/inception", inception_score, self.current_epoch)
        self.logger.experiment.add_scalar("metrics/lpips", lpips_score, self.current_epoch)
        
    def configure_optimizers(self):
        param_gen = list(self.img_encoder.parameters()) + list(self.img_decoder.parameters()) + list(self.quantize.parameters()) + \
            list(self.gs_embed.parameters() if self.gs_embed else [])
        param_disc = list(self.loss.discriminator.parameters())
        
        opt_gen = create_optimizer(param_gen, **self.optim_gen_cfg)
        opt_disc = create_optimizer(param_disc, **self.optim_disc_cfg)

        step_per_epoch  = len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size
        self.lr_scheduler_cfg["updates_per_epoch"] = step_per_epoch
        self.lr_scheduler_cfg["num_epochs"] = self.trainer.max_epochs

        scheduler_gen = create_scheduler(optimizer=opt_gen, **self.lr_scheduler_cfg)[0] # return scheduler, num_epochs
        scheduler_disc = create_scheduler(optimizer=opt_disc, **self.lr_scheduler_cfg)[0]
        
        return {"optimizer": opt_gen, "lr_scheduler": scheduler_gen}, {"optimizer": opt_disc, "lr_scheduler": scheduler_disc}