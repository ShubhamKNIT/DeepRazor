import os
import torch
import argparse
from env_var import *
from src.model.generator import *
from src.dataset.dataloader import *
from src.utils.notify_me import *
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

# Prepare the input image
def make_inference(generator, img, mask):
    coase_out, refine_out = None, None

    # add 1 as batch_size
    img = img.unsqueeze(0)
    mask = mask.unsqueeze(0)

    print(img.shape)
    print(mask.shape)

    with torch.no_grad():
        coarse_out, refine_out = generator(img, mask)

    coase_out = coarse_out.squeeze(0)
    refine_out = refine_out.squeeze(0)
    print(coase_out.shape)
    print(refine_out.shape)

    return coase_out, refine_out

def test(chkpt_no, val_loader, num_samples=2, save_imgs=True, chkpt_dir=CHKPT_DIR):
    chkpt_path = f'{chkpt_dir}/checkpoint_{chkpt_no}.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator()
    generator.to(device)

    # Load checkpoint
    checkpoint = torch.load(chkpt_path, map_location="cpu")
    generator.load_state_dict(checkpoint["gen_state_dict"], strict=False)
    generator.eval()

    # Use dataloader
    data = iter(val_loader)
    for i in range(num_samples):
        # Generate images
        real_images, masks, ground_truths = next(data)
        real_image, mask, ground_truth = real_images[0], masks[0], ground_truths[0]
        real_image, mask, ground_truth = real_image.to(device), mask.to(device), ground_truth.to(device)
        coarse_output, refined_output = make_inference(generator, real_image, mask)
        coarse_ = real_image * (1 - mask) + coarse_output * mask
        refine_ = real_image * (1 - mask) + refined_output * mask
        img_ = real_image * (1 - mask) 

        # Plot images
        fig, axs = plt.subplots(2, 4, figsize=(10, 8))

        axs[0][0].imshow(real_image.permute(1, 2, 0).cpu().numpy())
        axs[0][0].set_title('Input Image')
        axs[0][0].axis('off')

        axs[0][1].imshow(mask.squeeze().cpu().numpy(), cmap='gray')
        axs[0][1].set_title('Mask')
        axs[0][1].axis('off')

        axs[0][2].imshow(img_.permute(1, 2, 0).cpu().numpy())
        axs[0][2].set_title('Input-Infused')
        axs[0][2].axis('off')  

        axs[0][3].imshow(ground_truth.permute(1, 2, 0).cpu().numpy())
        axs[0][3].set_title('Ground Truth')
        axs[0][3].axis('off')

        axs[1][0].imshow(coarse_output.permute(1, 2, 0).cpu().numpy())
        axs[1][0].set_title('Coarse Output')
        axs[1][0].axis('off')
        
        axs[1][1].imshow(coarse_.permute(1, 2, 0).cpu().numpy())
        axs[1][1].set_title('Coarse-Infused Output')
        axs[1][1].axis('off')
        
        axs[1][2].imshow(refined_output.permute(1, 2, 0).cpu().numpy())
        axs[1][2].set_title('Refined Output')
        axs[1][2].axis('off')
        
        axs[1][3].imshow(refine_.permute(1, 2, 0).cpu().numpy())
        axs[1][3].set_title('Refined-Infused Output')
        axs[1][3].axis('off')

        if save_imgs:
            plt.savefig(f'{VAL_IMG_DIR}/epoch_{chkpt_no}_result_{i}.png')

        plt.tight_layout()
        plt.show()
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chkpt_no', type = int, default = -1, help = 'Enter a valid chkpt number')
    parser.add_argument('--num_chkpts', type = int, default = 1, help = 'Enter number of chkpt')
    parser.add_argument('--num_samples', type = int, default = 2, help = 'Number of Samples wanted')
    parser.add_argument('--img_size', type = int, default = 512, help = 'Image size')
    parser.add_argument('--save_imgs', type = bool, default = True, help = 'Save Image Enable/Disable')
    parser.add_argument('--notify', type = bool, default = False, help = "Send notification to telegram")
    opt = parser.parse_args()

    os.makedirs(f"{ROOT_DIR}/Results", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor()
    ])

    num_workers = os.cpu_count()
    val_dataset = ObjectRemovalDataset(data_dir=f"{DATA_DIR}", data_type="val", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=opt.num_samples, shuffle=True, num_workers=num_workers)
    # print(len(val_loader))
    for chkpt_no in range(opt.chkpt_no, opt.chkpt_no + opt.num_chkpts):
        test(chkpt_no, val_loader, num_samples=opt.num_samples, save_imgs=opt.save_imgs)  
        
    if opt.notify:
        send_notification(opt.chkpt_no, opt.num_chkpts, opt.num_samples, True, False, False, False)