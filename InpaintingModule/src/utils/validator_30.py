import gc
import torch
from tqdm import tqdm
from src.model.vgg import *
import torch.nn.functional as F
from torchmetrics.image import (
    PeakSignalNoiseRatio as PSNR,
    StructuralSimilarityIndexMeasure as SSIM,
    FrechetInceptionDistance as FID,
    LearnedPerceptualImagePatchSimilarity as LPIPS,
)

class Validator:
    def __init__(self, generator, discriminator, logger, device='cuda'):

        # Initailize
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.vgg = VGG16PerceptualFeatures().to(device)
        self.logger = logger
        self.device = device

        # Loss functions and weights as before
        self.lambda_l1 = 100
        self.lambda_perceptual = 10
        self.beta = 1
        self.sigma = 10
        self.epsilon = 1e-10
        self.lambda_fm = 10

        # Metrics
        self.psnr = PSNR().to(device)
        self.ssim = SSIM().to(device)
        self.fid = FID(feature=64).to(device)  # Use feature=64 for faster computation
        self.lpips = LPIPS(normalize=True).to(device)
    
    def validate(self, chkpt_no, val_loader):
        self.generator.eval()
        self.discriminator.eval()

        num_batches = 0
        epoch_d = {
            "coarse_loss": 0.0, "rec_loss": 0.0, "adv_loss": 0.0,
            "perceptual_loss": 0.0, "fm_loss": 0.0, "gen_loss": 0.0, "disc_loss": 0.0,
            "psnr": 0.0, "ssim": 0.0, "fid": 0.0, "lpips": 0.0
        }

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                images, masks, ground_truths = batch
                images, masks, ground_truths = images.to(self.device), masks.to(self.device), ground_truths.to(self.device)

                # Coarse Inpainting
                coarse_out, refine_out = self.generator(images, masks)
                coarse_out_ = images * (1 - masks) + coarse_out * masks
                refine_out_ = images * (1 - masks) + refine_out * masks

                # Discriminator Loss
                fake_images = refine_out_.detach()
                ground_preds = self.discriminator(ground_truths, masks)
                fake_preds = self.discriminator(fake_images, masks)
                # gp = self.compute_gradient_penalty(self.discriminator, ground_truths, fake_images, masks)
                d_loss_real = torch.mean(F.relu(1.0 - ground_preds))
                d_loss_fake = torch.mean(F.relu(1.0 + fake_preds))
                disc_loss = d_loss_real + d_loss_fake

                # Generator Loss
                coarse_loss = F.l1_loss(coarse_out_, ground_truths) + self.epsilon * 0.01
                refine_loss = F.l1_loss(refine_out_, ground_truths) + self.epsilon * 0.01
                adv_loss = -torch.mean(self.discriminator(refine_out_, masks)) + self.epsilon * 0.01
                perceptual_loss = F.l1_loss(self.vgg(ground_truths), self.vgg(refine_out_)) + self.epsilon * 0.01
                feature_match_loss = F.l1_loss(self.discriminator(ground_truths, masks), self.discriminator(refine_out_, masks))
                
                rec_loss = 0.5 * self.lambda_l1 * coarse_loss + self.lambda_l1 * refine_loss + self.epsilon * 0.01
                gen_loss = rec_loss + self.beta * adv_loss + self.lambda_perceptual * perceptual_loss + self.lambda_fm * feature_match_loss + self.epsilon * 0.01

                refine_out_uint8 = (refine_out * 255).byte()
                ground_truths_uint8 = (ground_truths * 255).byte()

                # Metrics
                psnr_value = self.psnr(refine_out_, ground_truths)
                ssim_value = self.ssim(refine_out_, ground_truths)
                self.fid.update(refine_out_uint8, real=False)
                self.fid.update(ground_truths_uint8, real=True)
                fid_value = self.fid.compute()
                self.fid.reset()
                lpips_value = self.lpips(refine_out_, ground_truths)

                # Accumulate losses
                epoch_d["coarse_loss"] += coarse_loss.item()
                epoch_d["rec_loss"] += rec_loss.item()
                epoch_d["adv_loss"] += adv_loss.item()
                epoch_d["perceptual_loss"] += perceptual_loss.item()
                epoch_d["fm_loss"] += feature_match_loss.item()
                epoch_d["gen_loss"] += gen_loss.item()
                epoch_d["disc_loss"] += disc_loss.item()
                epoch_d["psnr"] += psnr_value.item()
                epoch_d["ssim"] += ssim_value.item()
                epoch_d["fid"] += fid_value.item()
                epoch_d["lpips"] += lpips_value.item()
                num_batches += 1

                torch.cuda.empty_cache()
                gc.collect()

        self.generator.train()
        self.discriminator.train()

        # Average losses and metrics
        for key in epoch_d:
            epoch_d[key] /= num_batches

        # Log validation results
        self.logger.log_epoch(chkpt_no, epoch_d)

        print(f"Validation - Epoch {chkpt_no}: "
              f"Coarse Loss: {epoch_d['coarse_loss']:.4f}, "
              f"Reconstruction Loss: {epoch_d['rec_loss']:.4f}, "
              f"Adversarial Loss: {epoch_d['adv_loss']:.4f}, "
              f"Perceptual Loss: {epoch_d['perceptual_loss']:.4f}, "
              f"Feature Match Loss: {epoch_d['fm_loss']:.4f}, "
              f"Generator Loss: {epoch_d['gen_loss']:.4f}, "
              f"Discriminator Loss: {epoch_d['disc_loss']:.4f}, "
              f"PSNR: {epoch_d['psnr']:.4f}, "
              f"SSIM: {epoch_d['ssim']:.4f}, "
              f"FID: {epoch_d['fid']:.4f}, "
              f"LPIPS: {epoch_d['lpips']:.4f}")

        self.generator.train()
        self.discriminator.train()