import gc
import torch
import time
from tqdm import tqdm
from src.model.vgg import VGG16PerceptualFeatures
import torch.nn.functional as F
from torchmetrics.image import (
    PeakSignalNoiseRatio as PSNR,
    StructuralSimilarityIndexMeasure as SSIM,
    FrechetInceptionDistance as FID,
    LearnedPerceptualImagePatchSimilarity as LPIPS,
)
import torch.fft

class Validator:
    def __init__(self, generator, discriminator, logger, device='cuda'):
        # Initialize models and utilities
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.vgg = VGG16PerceptualFeatures().to(device)
        self.logger = logger
        self.device = device

        # Loss weights (should match those used during training)
        # self.lambda_l1 = 100            
        # self.lambda_perceptual = 10     
        # self.lambda_tv = 10             
        # self.lambda_adv = 1             
        # self.lambda_fft = 10           

        self.lambda_l1 = 100            # weight for L1 loss
        self.lambda_perceptual = 10      # weight for perceptual (VGG) loss in refinement
        self.lambda_tv = 10             # weight for Total Variation loss in coarse output
        self.lambda_adv = 1             # weight for adversarial loss in refinement
        self.lambda_fft = 5            # weight for Fourier loss (common to both stages)
        self.epsilon = 1e-10
        
        # Metrics
        self.psnr = PSNR().to(device)
        self.ssim = SSIM().to(device)
        self.fid = FID(feature=64).to(device)  # Using feature=64 for speed
        self.lpips = LPIPS(normalize=True).to(device)
    
    def compute_fourier_loss(self, pred, target):
        # Create a Hann window to reduce edge artifacts
        B, C, H, W = pred.shape
        hann_y = torch.hann_window(H, device=pred.device).unsqueeze(1)
        hann_x = torch.hann_window(W, device=pred.device).unsqueeze(0)
        window = hann_y @ hann_x  # [H, W]
        window = window.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)
        
        pred_win = pred * window
        target_win = target * window

        # Compute FFT and shift zero-frequency to center
        fft_pred = torch.fft.fftshift(torch.fft.fft2(pred_win, norm='ortho'))
        fft_target = torch.fft.fftshift(torch.fft.fft2(target_win, norm='ortho'))
        
        # Compute amplitude and phase
        amp_pred = torch.abs(fft_pred)
        amp_target = torch.abs(fft_target)
        phase_pred = torch.angle(fft_pred)
        phase_target = torch.angle(fft_target)
        
        loss_amp = F.l1_loss(amp_pred, amp_target)
        loss_phase = F.l1_loss(phase_pred, phase_target)
        return loss_amp + loss_phase
    def compute_tv_loss(self, img):
        # Total Variation (TV) loss for smoothness
        tv_y = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
        tv_x = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
        return tv_y + tv_x

    def validate(self, chkpt_no, val_loader):
        self.generator.eval()
        self.discriminator.eval()

        num_batches = 0
        epoch_d = {
            "coarse_loss": 0.0, "rec_loss": 0.0, "adv_loss": 0.0, 
            "vgg_loss": 0.0, "gen_loss": 0.0, "disc_loss": 0.0,
            "fourier_loss_coarse": 0.0, "fourier_loss_refine": 0.0,
            "tv_loss": 0.0, "psnr": 0.0, "ssim": 0.0,
            "fid": 0.0, "lpips": 0.0,
        }

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                images, masks, ground_truths = batch
                images = images.to(self.device)
                masks = masks.to(self.device)
                ground_truths = ground_truths.to(self.device)

                # Forward pass through generator
                coarse_out, refine_out = self.generator(images, masks)
                coarse_out_ = images * (1 - masks) + coarse_out * masks
                refine_out_ = images * (1 - masks) + refine_out * masks

                # Discriminator Loss (Spatial Domain)
                fake_images = refine_out_.detach()
                ground_preds = self.discriminator(ground_truths, masks)
                fake_preds = self.discriminator(fake_images, masks)
                d_loss_real = torch.mean(F.relu(1.0 - ground_preds))
                d_loss_fake = torch.mean(F.relu(1.0 + fake_preds))
                disc_loss = d_loss_real + d_loss_fake

                # Coarse Generator Loss
                L_l1_coarse = F.l1_loss(coarse_out_, ground_truths)
                L_F_coarse = self.compute_fourier_loss(coarse_out_, ground_truths)
                # L_adv_coarse = -torch.mean(self.discriminator(coarse_out_, masks))
                L_vgg_coarse = F.l1_loss(self.vgg(ground_truths), self.vgg(coarse_out_))
                L_TV = self.compute_tv_loss(coarse_out_)
                L_coarse = self.lambda_l1 * L_l1_coarse + self.lambda_fft * L_F_coarse + self.lambda_tv * L_TV

                # --- Refinement Generator Loss --- #
                L_l1_refine = F.l1_loss(refine_out_, ground_truths)
                L_l1_coarse_for_refine = F.l1_loss(coarse_out_, ground_truths)
                L_F_refine = self.compute_fourier_loss(refine_out_, ground_truths)
                L_adv_refine = -torch.mean(self.discriminator(refine_out_, masks))
                L_vgg_refine = F.l1_loss(self.vgg(ground_truths), self.vgg(refine_out_))
                L_rec_refine = (L_l1_refine + L_l1_coarse_for_refine) / 2
                L_refine = self.lambda_l1 * L_rec_refine + self.lambda_fft * L_F_refine + self.lambda_adv * L_adv_refine + self.lambda_perceptual * L_vgg_refine

                # Total generator loss
                gen_loss = L_coarse + L_refine

                # Metrics
                psnr_value = self.psnr(refine_out_, ground_truths)
                ssim_value = self.ssim(refine_out_, ground_truths)
                # For FID and LPIPS, cast images to uint8 if needed:
                refine_out_uint8 = (refine_out_ * 255).clamp(0, 255).to(torch.uint8)
                ground_truths_uint8 = (ground_truths * 255).clamp(0, 255).to(torch.uint8)
                self.fid.update(refine_out_uint8, real=False)
                self.fid.update(ground_truths_uint8, real=True)
                fid_value = self.fid.compute()
                self.fid.reset()
                lpips_value = self.lpips(refine_out_, ground_truths)

                # Accumulate losses/metrics
                epoch_d["coarse_loss"] += L_coarse.item()
                epoch_d["rec_loss"] += L_rec_refine.item()
                epoch_d["adv_loss"] += L_adv_refine.item()
                epoch_d["vgg_loss"] += (L_vgg_coarse + L_vgg_refine).item()
                epoch_d["gen_loss"] += gen_loss.item()
                epoch_d["disc_loss"] += disc_loss.item()
                epoch_d["fourier_loss_coarse"] += L_F_coarse.item()
                epoch_d["fourier_loss_refine"] += L_F_refine.item()
                epoch_d["tv_loss"] += L_TV.item()
                epoch_d["psnr"] += psnr_value.item()
                epoch_d["ssim"] += ssim_value.item()
                epoch_d["fid"] += fid_value.item()
                epoch_d["lpips"] += lpips_value.item()
                num_batches += 1

                torch.cuda.empty_cache()
                gc.collect()

        # Average losses and metrics over batches
        for key in epoch_d:
            epoch_d[key] /= num_batches

        # Log the validation results
        self.logger.log_epoch(chkpt_no, epoch_d)
        print(f"Validation - Epoch {chkpt_no}: "
              f"Coarse Loss: {epoch_d['coarse_loss']:.4f}, "
              f"Rec Loss: {epoch_d['rec_loss']:.4f}, "
              f"Adv Loss: {epoch_d['adv_loss']:.4f}, "
              f"VGG Loss: {epoch_d['vgg_loss']:.4f}, "
              f"TV Loss: {epoch_d['tv_loss']:.4f}, "
              f"Gen Loss: {epoch_d['gen_loss']:.4f}, "
              f"Disc Loss: {epoch_d['disc_loss']:.4f}, "
              f"PSNR: {epoch_d['psnr']:.4f}, "
              f"SSIM: {epoch_d['ssim']:.4f}, "
              f"FID: {epoch_d['fid']:.4f}, "
              f"LPIPS: {epoch_d['lpips']:.4f}")
    
        self.generator.train()
        self.discriminator.train()