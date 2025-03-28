import sys
import os

# Add ROOT_DIR to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from env_var import *  # Now it should work
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.notify_me import *
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
        ('percept_loss', 'Perceptual Loss'),
        ('fm_loss', 'Feature Match Loss'),
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
    fig, ax = plt.subplots(3, 4, figsize=(12, 9))
    
    # Plot training vs validation metrics
    for i, (key, title) in enumerate(train_metrics):
        row, col = divmod(i, 4)
        ax[row][col].plot(train_df['epoch'], train_df[key], label=f'Train {title}', color='blue')
        ax[row][col].plot(val_df['epoch'], val_df[key], label=f'Val {title}', color='red', linestyle='dashed')
        ax[row][col].set_title(title)
        ax[row][col].grid(True)
        ax[row][col].legend()
    
    # Plot validation-only metrics
    for i, (key, title) in enumerate(val_metrics, start=len(train_metrics)):
        row, col = divmod(i, 4)
        ax[row][col].plot(val_df['epoch'], val_df[key], label=f'Val {title}', color='red')
        ax[row][col].set_title(title)
        ax[row][col].grid(True)
        ax[row][col].legend()
    
    # Learning rates
    ax[2][3].plot(train_df['epoch'], train_df['gen_lr'], label='Generator LR', color='blue')
    ax[2][3].plot(train_df['epoch'], train_df['disc_lr'], label='Discriminator LR', color='red', linestyle='dashed')
    ax[2][3].set_title('Generator vs Discriminator LR')
    ax[2][3].grid(True)
    ax[2][3].legend()
    
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