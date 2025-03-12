import sys
import os

# Get the absolute path to ROOT_DIR (Two levels up)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Add ROOT_DIR to sys.path
sys.path.append(ROOT_DIR)

from env_var import *  # Now it can find env_var.py
from src.utils.notify_me import *  # Import from utils
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_comparision(notify):
    # Load CSVs
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    val_df = pd.read_csv(VAL_CSV_PATH)
    
    # Save cleaned CSVs
    train_df.to_csv(TRAIN_CSV_PATH, index=False)
    val_df.to_csv(VAL_CSV_PATH, index=False)
    
    # Define plot titles and labels
    train_metrics = [
        ('coarse_loss', 'Coarse Loss'),
        ('rec_loss', 'Reconstruction Loss'),
        ('adv_loss', 'Adversarial Loss'),
        ('vgg_loss', 'Perceptual Loss'),
        ('fourier_loss_coarse', 'Fourier Coarse Loss'),
        ('fourier_loss_refine', 'Fourier Refine Loss'),
        ('tv_loss', 'TransVariation Loss'),
        ('gen_loss', 'Generator Loss'),
        ('disc_loss', 'Discriminator Loss')
    ]
    
    val_metrics = [
        ('psnr', 'PSNR'),
        ('ssim', 'SSIM'),
        ('fid', 'FID'),
        ('lpips', 'LPIPS')
    ]
    
    # Create subplots
    fig, ax = plt.subplots(5, 3, figsize=(9, 15))
    
    # Plot training vs validation metrics
    for i, (key, title) in enumerate(train_metrics):
        row, col = divmod(i, 3)
        ax[row][col].plot(train_df['epoch'], train_df[key], label=f'Train {title}', color='blue')
        ax[row][col].plot(val_df['epoch'], val_df[key], label=f'Val {title}', color='red', linestyle='dashed')
        ax[row][col].set_title(title)
        ax[row][col].grid(True)
        ax[row][col].legend()
    
    # Plot validation-only metrics
    for i, (key, title) in enumerate(val_metrics, start=len(train_metrics)):
        row, col = divmod(i, 3)
        ax[row][col].plot(val_df['epoch'], val_df[key], label=f'Val {title}', color='red')
        ax[row][col].set_title(title)
        ax[row][col].grid(True)
        ax[row][col].legend()
    
    # Learning rates
    ax[4][2].plot(train_df['epoch'], train_df['gen_lr'], label='Generator LR', color='blue')
    ax[4][2].plot(train_df['epoch'], train_df['disc_lr'], label='Discriminator LR', color='red', linestyle='dashed')
    ax[4][2].set_title('Generator vs Discriminator LR')
    ax[4][2].grid(True)
    ax[4][2].legend()
    
    fig.tight_layout()
    fig.savefig(f'{VAL_IMG_DIR}/comparison_plot.png')
    plt.show()

    if notify:
        send_notification(0, 0, 0, False, False, False, True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--notify', type = bool, default = True, help = 'Send Comparision chart to telegram')
    opt = parser.parse_args()
    plot_comparision(opt.notify)