import numpy as np
import matplotlib.pyplot as plt
from src.dataset.dataloader import *
from env_var import *

def visualize_samples(dataloader, num_samples=2):
    data_iter = iter(dataloader)
    input_data, mask_data, gt_data = next(data_iter)  # First batch

    for i in range(num_samples):
        input_sample = input_data[i]
        mask_sample = mask_data[i]
        gt_sample = gt_data[i]
        infused_sample = input_sample * (1 - mask_sample)
        # print(mask_sample)

        # Convert tensors to numpy arrays
        input_sample_np = input_sample.permute(1, 2, 0).cpu().numpy()
        mask_sample_np = mask_sample.squeeze(0).cpu().numpy()  # Remove extra dimensions
        gt_sample_np = gt_sample.permute(1, 2, 0).cpu().numpy()
        infused_sample_np = infused_sample.permute(1, 2, 0).cpu().numpy()

        # Rescale the image tensors from [0, 1] to [0, 255] for visualization
        input_sample_np = np.clip(input_sample_np * 255, 0, 255).astype(np.uint8)
        gt_sample_np = np.clip(gt_sample_np * 255, 0, 255).astype(np.uint8)
        infused_sample_np = np.clip(infused_sample_np * 255, 0, 255).astype(np.uint8)

        # Normalize mask values to binary (0 and 255)
        mask_sample_np = np.clip(mask_sample_np * 255, 0, 255).astype(np.uint8)
        mask_sample_np = np.where(mask_sample_np > 128, 255, 0)  # Set mask to 0 or 255

        # Plot the images
        plt.figure(figsize=(12, 12))

        # Display input image
        plt.subplot(1, 4, 1)
        plt.imshow(input_sample_np)
        plt.title("Image")
        plt.axis("off")

        # Display mask
        plt.subplot(1, 4, 2)
        plt.imshow(mask_sample_np, cmap="gray")
        plt.title("Mask")
        plt.axis("off")

        # Display ground truth
        plt.subplot(1, 4, 3)
        plt.imshow(gt_sample_np)
        plt.title("Ground Truth")
        plt.axis("off")

        # Display infused image (mask applied on input)
        plt.subplot(1, 4, 4)
        plt.imshow(infused_sample_np)
        plt.title("Infused Image")
        plt.axis("off")
        plt.save_fig(f"{VAL_IMG_DIR}/sample_{i}")
        plt.show()


if __name__ == "__main__":
    """
        Copy and past the code below in Jupyter Notebook
    """
    # from src.utils.visualize_samples import *

    num_samples = 2
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    num_workers = os.cpu_count()
    val_dataset = ObjectRemovalDataset(data_dir=f"{DATA_DIR}", data_type="val", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=num_workers)
    visualize_samples(val_loader)