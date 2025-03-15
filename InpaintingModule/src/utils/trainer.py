import gc
import torch
import time
from tqdm import tqdm
from src.model.vgg import VGG16PerceptualFeatures
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import torch.fft

class Trainer:
    def __init__(self, generator, discriminator, 
                 opt_c, opt_r, opt_d,
                 sched_c, sched_r, sched_d,
                 dataloader, logger, device='cuda'):
        # Initialize models and utilities
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.vgg = VGG16PerceptualFeatures().to(device)
        self.dataloader = dataloader
        self.logger = logger
        self.device = device

        # Optimizers
        self.opt_c = opt_c
        self.opt_r = opt_r
        self.opt_d = opt_d

        # Schedulers
        self.sched_c = sched_c
        self.sched_r = sched_r
        self.sched_d = sched_d

        # GradScalers for AMP
        self.scaler_c = GradScaler()
        self.scaler_r = GradScaler()
        self.scaler_d = GradScaler()
        
        # Loss weights (hyperparameters)
        # self.lambda_l1 = 100            # weight for L1 loss
        # self.lambda_perceptual = 10      # weight for perceptual (VGG) loss in refinement
        # self.lambda_tv = 10             # weight for Total Variation loss in coarse output
        # self.lambda_adv = 1             # weight for adversarial loss in refinement
        # self.lambda_fft = 10            # weight for Fourier loss (common to both stages)

        self.lambda_l1 = 100            # weight for L1 loss
        self.lambda_perceptual = 10      # weight for perceptual (VGG) loss in refinement
        self.lambda_tv = 10             # weight for Total Variation loss in coarse output
        self.lambda_adv = 1             # weight for adversarial loss in refinement
        self.lambda_fft = 5            # weight for Fourier loss (common to both stages)
        self.epsilon = 1e-10

        # NaN stopping criteria
        self.nan_counter = 0
        self.max_nan_count = 5

    def check_for_nans(self, model):
        for param in model.parameters():
            if torch.isnan(param).any():
                return True
        return False

    def compute_gradient_penalty(self, discriminator, real_samples, fake_samples, masks):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(self.device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        disc_interpolates = discriminator(interpolates, masks)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        return gradient_penalty

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
        return (loss_amp + loss_phase)/2

    def compute_tv_loss(self, img):
        # Total Variation (TV) loss for smoothness
        tv_y = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
        tv_x = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
        return (tv_y + tv_x)/2

    def train_step(self, batch, step_count):
        images, masks, ground_truths = batch
        images, masks, ground_truths = images.to(self.device), masks.to(self.device), ground_truths.to(self.device)
    
        max_norm = 5.0  # Clipping threshold
    
        with autocast(self.device):
            # Generator forward pass: obtain coarse and refined outputs
            coarse_out, refine_out = self.generator(images, masks)
            coarse_out_ = images * (1 - masks) + coarse_out * masks
            refine_out_ = images * (1 - masks) + refine_out * masks
    
            # --- Discriminator Loss (Spatial Domain) --- #
            self.opt_d.zero_grad()
            fake_images = refine_out_.detach()
            ground_preds = self.discriminator(ground_truths.detach(), masks)
            fake_preds = self.discriminator(fake_images, masks)
            d_loss_real = torch.mean(F.relu(1.0 - ground_preds))
            d_loss_fake = torch.mean(F.relu(1.0 + fake_preds))
            disc_loss = d_loss_real + d_loss_fake
            # Optionally add gradient penalty if needed:
            # gp = self.compute_gradient_penalty(self.discriminator, ground_truths, fake_images, masks)
            # disc_loss += self.sigma * gp + self.epsilon
            
            self.scaler_d.scale(disc_loss).backward()
            self.scaler_d.unscale_(self.opt_d)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm)
            self.scaler_d.step(self.opt_d)
            self.scaler_d.update()
    
            # --- Coarse Generator Loss --- #
            self.opt_c.zero_grad()
            L_l1_coarse = F.l1_loss(coarse_out_, ground_truths)
            L_F_coarse = self.compute_fourier_loss(coarse_out_, ground_truths)
            L_TV = self.compute_tv_loss(coarse_out_)
            # Note: We drop adversarial loss in the coarse stage per the new formulation.
            L_coarse = self.lambda_l1 * L_l1_coarse + self.lambda_fft * L_F_coarse + self.lambda_tv * L_TV
    
            self.scaler_c.scale(L_coarse).backward(retain_graph=True)
            self.scaler_c.unscale_(self.opt_c)
            torch.nn.utils.clip_grad_norm_(self.generator.coarse_generator.parameters(), max_norm)
            self.scaler_c.step(self.opt_c)
            self.scaler_c.update()
    
            # --- Refinement Generator Loss --- #
            self.opt_r.zero_grad()
            L_l1_refine = F.l1_loss(refine_out_, ground_truths)
            L_F_refine = self.compute_fourier_loss(refine_out_, ground_truths)
            L_adv_refine = -torch.mean(self.discriminator(refine_out_, masks))
            L_vgg_refine = F.l1_loss(self.vgg(ground_truths), self.vgg(refine_out_))
    
            # Reconstruction loss for refinement: average L1 loss (refine and coarse) plus Fourier loss
            L_rec_refine = (L_l1_refine + L_l1_coarse) / 2
            L_refine = self.lambda_l1 * L_rec_refine + self.lambda_fft * L_F_refine + self.lambda_adv * L_adv_refine + self.lambda_perceptual * L_vgg_refine
    
            self.scaler_r.scale(L_refine).backward()
            self.scaler_r.unscale_(self.opt_r)
            torch.nn.utils.clip_grad_norm_(self.generator.refine_generator.parameters(), max_norm)
            self.scaler_r.step(self.opt_r)
            self.scaler_r.update()
    
            return {
                "coarse_loss": L_coarse.item(),
                "rec_loss": L_rec_refine.item(),
                "adv_loss": L_adv_refine.item(),
                "vgg_loss": L_vgg_refine.item(),
                "fourier_loss_coarse": L_F_coarse.item(),
                "fourier_loss_refine": L_F_refine.item(),
                "tv_loss": L_TV.item(),
                "gen_loss": (L_coarse + L_refine).item(),
                "disc_loss": disc_loss.item(),
            }
    
    def train(self, start_epoch, num_epochs):
        for epoch in range(start_epoch, start_epoch + num_epochs):
            start_time = time.time()
            epoch_d = {
                "coarse_loss": 0.0, "rec_loss": 0.0, "adv_loss": 0.0,
                "vgg_loss": 0.0, "fourier_loss_coarse": 0.0, 
                "fourier_loss_refine": 0.0, "tv_loss": 0.0, 
                "gen_loss": 0.0, "disc_loss": 0.0,
            }
    
            print(f"Epoch [{epoch}/{start_epoch + num_epochs - 1}] - Learning Rates:")
            print(f"Generator LR: {self.opt_r.param_groups[0]['lr']}")
            print(f"Discriminator LR: {self.opt_d.param_groups[0]['lr']}")
    
            for step_count, batch in enumerate(tqdm(self.dataloader, desc=f"Epoch {epoch}/{start_epoch + num_epochs - 1}", dynamic_ncols=True)):
                batch_loss = self.train_step(batch, step_count)
                for key, value in batch_loss.items():
                    epoch_d[key] += value
    
            training_time = (time.time() - start_time) / 60
            for key in epoch_d:
                epoch_d[key] /= len(self.dataloader)
    
            curr_gen_lr = self.opt_c.param_groups[0]['lr']
            curr_disc_lr = self.opt_d.param_groups[0]['lr']
    
            # Update learning rate schedulers (if needed)
            # self.sched_c.step()
            # self.sched_r.step()
            # self.sched_d.step()
    
            epoch_d['gen_lr'] = curr_gen_lr
            epoch_d['disc_lr'] = curr_disc_lr
    
            self.logger.log_epoch(epoch, epoch_d)
            self.logger.save_checkpoint(epoch, self.generator, self.discriminator,
                                        self.opt_c, self.opt_r, self.opt_d,
                                        self.sched_c, self.sched_r, self.sched_d)
    
            print(f"Epoch [{epoch}/{start_epoch + num_epochs - 1}] - "
                  f"Coarse Loss: {epoch_d['coarse_loss']:.4f}, "
                  f"Rec Loss: {epoch_d['rec_loss']:.4f}, "
                  f"Adv Loss: {epoch_d['adv_loss']:.4f}, "
                  f"VGG Loss: {epoch_d['vgg_loss']:.4f}, "
                  f"Fourier Loss (Coarse): {epoch_d['fourier_loss_coarse']:.4f}, "
                  f"Fourier Loss (Refine): {epoch_d['fourier_loss_refine']:.4f}, "
                  f"TV Loss: {epoch_d['tv_loss']:.4f}, "
                  f"Gen Loss: {epoch_d['gen_loss']:.4f}, "
                  f"Disc Loss: {epoch_d['disc_loss']:.4f}, "
                  f"Gen LR: {curr_gen_lr:.6f}, "
                  f"Disc LR: {curr_disc_lr:.6f}, "
                  f"Training Time: {training_time:.2f} m")
    
            print(f"GPU Memory Usage: {torch.cuda.max_memory_allocated(device=self.device)/(1024**2)} MB")
            torch.cuda.empty_cache()
            gc.collect()