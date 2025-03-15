import gc
import torch
import time
from tqdm import tqdm
from src.model.vgg import *
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

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
        
        # Loss functions and weights
        self.lambda_l1 = 100
        self.lambda_perceptual = 10
        self.beta = 1
        self.sigma = 3
        self.epsilon = 1e-10
        self.lambda_fm = 10

        # NaN stopping criteria
        self.nan_counter = 0
        self.max_nan_count = 5

    def check_for_nans(self, model):  # To avoid exploding gradients and stop
        for param in model.parameters():
            if torch.isnan(param).any():
                return True
        return False

    def compute_gradient_penalty(self, discriminator, real_samples, fake_samples, masks):
        # Compute gradient penalty as before
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
            
    def train_step(self, batch, step_count):
        images, masks, ground_truths = batch
        images, masks, ground_truths = images.to(self.device), masks.to(self.device), ground_truths.to(self.device)
    
        max_norm = 5.0  # Increased clipping threshold
    
        with autocast(self.device):
            coarse_out, refine_out = self.generator(images, masks)
            coarse_out_ = images * (1 - masks) + coarse_out * masks
            refine_out_ = images * (1 - masks) + refine_out * masks
    
            # Zero gradients
            self.opt_d.zero_grad()
            
            # Discriminator Loss
            fake_images = refine_out_.detach()
            ground_preds = self.discriminator(ground_truths.detach(), masks)
            fake_preds = self.discriminator(fake_images, masks)
            d_loss_real = torch.mean(F.relu(1.0 - ground_preds))
            d_loss_fake = torch.mean(F.relu(1.0 + fake_preds))
            disc_loss = d_loss_real + d_loss_fake
    
            self.scaler_d.scale(disc_loss).backward()
            self.scaler_d.unscale_(self.opt_d)  # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm)
            self.scaler_d.step(self.opt_d)
            self.scaler_d.update()
    
            # Coarse Generator Loss
            self.opt_c.zero_grad()
            coarse_loss = F.l1_loss(coarse_out_, ground_truths) + self.epsilon * 0.01
            
            self.scaler_c.scale(coarse_loss).backward(retain_graph=True)
            self.scaler_c.unscale_(self.opt_c)  # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(self.generator.coarse_generator.parameters(), max_norm)
            self.scaler_c.step(self.opt_c)
            self.scaler_c.update()            
    
            # Refinement Generator Loss
            self.opt_r.zero_grad()
            refine_loss = F.l1_loss(refine_out_, ground_truths) + self.epsilon * 0.01
            adv_loss = -torch.mean(self.discriminator(refine_out_, masks)) + self.epsilon * 0.01
            perceptual_loss = F.l1_loss(self.vgg(ground_truths), self.vgg(refine_out_)) + self.epsilon * 0.01
            feature_match_loss = F.l1_loss(self.discriminator(ground_truths.detach(), masks), self.discriminator(refine_out_, masks))
    
            rec_loss = 0.5 * self.lambda_l1 * coarse_loss + self.lambda_l1 * refine_loss + self.epsilon * 0.01
            gen_loss = rec_loss + self.beta * adv_loss + self.lambda_perceptual * perceptual_loss + self.lambda_fm * feature_match_loss + self.epsilon * 0.01
    
            self.scaler_r.scale(gen_loss).backward()
            self.scaler_r.unscale_(self.opt_r)  # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(self.generator.refine_generator.parameters(), max_norm)
            self.scaler_r.step(self.opt_r)
            self.scaler_r.update()
    
            return {
                "coarse_loss": coarse_loss.item(),
                "rec_loss": rec_loss.item(),
                "adv_loss": adv_loss.item(),
                "perceptual_loss": perceptual_loss.item(),
                "fm_loss": feature_match_loss.item(),
                "gen_loss": gen_loss.item(),
                "disc_loss": disc_loss.item(),
            }


    def train(self, start_epoch, num_epochs):
        for epoch in range(start_epoch, start_epoch + num_epochs):
            start_time = time.time()
            epoch_d = {"coarse_loss": 0.0, "rec_loss": 0.0, "adv_loss": 0.0, "fm_loss": 0.0,
                       "perceptual_loss": 0.0, "gen_loss": 0, "disc_loss": 0.0}

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
            
            # Update learning rate schedulers
            # self.sched_c.step()
            # self.sched_r.step()
            self.sched_d.step()

            epoch_d['gen_lr'] = curr_gen_lr
            epoch_d['disc_lr'] = curr_disc_lr

            self.logger.log_epoch(epoch, epoch_d)
            self.logger.save_checkpoint(epoch, self.generator, self.discriminator, 
                                        self.opt_c, self.opt_r, self.opt_d,
                                        self.sched_c, self.sched_r, self.sched_d)

            print(f"Epoch [{epoch}/{start_epoch + num_epochs - 1}] - "
                f"Coarse Loss: {epoch_d['coarse_loss']:.4f}, "
                f"Reconstruction Loss: {epoch_d['rec_loss']:.4f}, "
                f"Adversarial Loss: {epoch_d['adv_loss']:.4f}, "
                f"Perceptual Loss: {epoch_d['perceptual_loss']:.4f}, "
                f"Feature Match Loss: {epoch_d['fm_loss']:.4f}, "
                f"Generator Loss: {epoch_d['gen_loss']:.4f}, "
                f"Discriminator Loss: {epoch_d['disc_loss']:.4f}, "
                f"Generator LR: {curr_gen_lr:.6f}, "
                f"Discriminator LR: {curr_disc_lr:.6f}, "
                f"Training Time: {training_time:.2f} m")


            print(f"GPU Memory Usage: {torch.cuda.max_memory_allocated(device=self.device)/(1024**2)} MB")
            torch.cuda.empty_cache()
            gc.collect()