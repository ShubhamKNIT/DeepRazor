import csv
import os
import torch

class Logger:
    def __init__(self, log_file, ftype, checkpoint_dir):
        """Initialize the logger with paths for CSV and checkpoint directories."""

        self.ftype = ftype
        self.log_file = log_file
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        if self.ftype == 'train':
            self._initialize_csv(self.log_file, [
                'epoch', 'coarse_loss', 'rec_loss', 'adv_loss', 'vgg_loss',
                'fourier_loss_coarse', 'fourier_loss_refine', 'tv_loss',
                'gen_loss', 'disc_loss', 'gen_lr', 'disc_lr'
            ])
        elif self.ftype == 'val':
            self._initialize_csv(self.log_file, [
                'epoch', 'coarse_loss', 'rec_loss', 'adv_loss', 'vgg_loss',
                'fourier_loss_coarse', 'fourier_loss_refine', 'tv_loss',
                'gen_loss', 'disc_loss', 'psnr', 'ssim', 'fid', 'lpips'
            ])
        else:
            raise ValueError(f"Invalid ftype: {self.ftype}. Must be 'train' or 'val'.")


    @staticmethod
    def _initialize_csv(file_path, headers):
        """Initialize a CSV file with headers if it doesn't exist."""
        if not os.path.exists(file_path):
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                print(f"New log file created/initialized at {file_path}")

    @staticmethod
    def _log_to_csv(file_path, row):
        """Append a row to the CSV file."""
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

    def log_epoch(self, epoch, epoch_d):
        """Log training metrics for an epoch."""
        row = [epoch]
        row += epoch_d.values()
        if self.ftype == 'train':
            self._log_to_csv(self.log_file, row)
        elif self.ftype == 'val':
            self._log_to_csv(self.log_file, row)
        else:
            raise ValueError("Invalid file type. Use 'train' or 'val'.")

    def save_checkpoint(self, epoch, gen, disc, 
                        opt_coarse, opt_refine, opt_disc,
                        sched_coarse, sched_refine, sched_disc):
        """Save a model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'gen_state_dict': gen.state_dict(),
            'disc_state_dict': disc.state_dict(),
            'opt_refine_state_dict': opt_refine.state_dict(),
            'opt_coarse_state_dict': opt_coarse.state_dict(),
            'opt_disc_state_dict': opt_disc.state_dict(),
            'sched_refine_state_dict': sched_refine.state_dict(),
            'sched_coarse_state_dict': sched_coarse.state_dict(),
            'sched_disc_state_dict': sched_disc.state_dict()
        }

        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, epoch, gen, disc,
                        opt_coarse=None, opt_refine=None, opt_disc=None,
                        sched_coarse=None, sched_refine=None, sched_disc=None):
        """Load a model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{epoch}.pth")
        checkpoint = torch.load(checkpoint_path)

        gen.load_state_dict(checkpoint['gen_state_dict'])
        disc.load_state_dict(checkpoint['disc_state_dict'])

        if opt_coarse:
            opt_refine.load_state_dict(checkpoint['opt_refine_state_dict'])
            opt_coarse.load_state_dict(checkpoint['opt_coarse_state_dict'])
            sched_refine.load_state_dict(checkpoint['sched_refine_state_dict'])
            sched_coarse.load_state_dict(checkpoint['sched_coarse_state_dict'])
        if opt_disc:
            opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])
            sched_disc.load_state_dict(checkpoint['sched_disc_state_dict'])
        print(f"Checkpoint loaded from {checkpoint_path}")

    # def load_checkpoint(self, epoch, gen, disc,
    #                     opt_coarse=None, opt_refine=None, opt_disc=None,
    #                     sched_coarse=None, sched_refine=None, sched_disc=None):
    #     """Load a model checkpoint."""
    #     checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{epoch}.pth")
    #     checkpoint = torch.load(checkpoint_path)
    
    #     # Load generator state dict with strict=False to ignore missing keys (like new swin_bottleneck)
    #     # missing_keys, unexpected_keys = gen.load_state_dict(checkpoint['gen_state_dict'], strict=False)
    #     # print("Generator missing keys:", missing_keys)
    #     # print("Generator unexpected keys:", unexpected_keys)
        
    #     gen.load_state_dict(checkpoint['gen_state_dict'], strict=False)
    #     disc.load_state_dict(checkpoint['disc_state_dict'])  # Assuming discriminator matches exactly
    
    #     # Load optimizer and scheduler state dicts (they don't accept strict)
    #     if opt_coarse is not None:
    #         opt_coarse.load_state_dict(checkpoint['opt_coarse_state_dict'])
    #         sched_coarse.load_state_dict(checkpoint['sched_coarse_state_dict'])
    #     if opt_refine is not None:
    #         opt_refine.load_state_dict(checkpoint['opt_refine_state_dict'], strict = False)
    #         sched_refine.load_state_dict(checkpoint['sched_refine_state_dict'], strict = False)
    #     if opt_disc is not None:
    #         opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])
    #         sched_disc.load_state_dict(checkpoint['sched_disc_state_dict'])
            
    #     print(f"Checkpoint loaded from {checkpoint_path}")
