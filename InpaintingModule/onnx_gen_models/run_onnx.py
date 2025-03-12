import sys
sys.path.append("/home/jupyter/InpaintingModule")

import os
import torch
import argparse
import numpy as np
import onnxruntime as ort
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from src.dataset.dataloader import ObjectRemovalDataset
from time import time

def tensor_to_image(tensor):
    np_img = tensor.cpu().numpy()
    np_img = np.clip(np_img * 255, 0, 255).astype(np.uint8)
    return np.transpose(np_img, (1, 2, 0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chkpt_no', type=int, default=59, help='Checkpoint number')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--num_batches', type=int, default=5, help='Number of batches to process')
    parser.add_argument('--img_size', type=int, default=512, help='Image size')
    parser.add_argument('--data_dir', type=str, default='path/to/data', help='Dataset directory')
    parser.add_argument('--save_folder', type=str, default='Results', help='Folder to save results')
    opt = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder = os.path.join(opt.save_folder, timestamp)
    os.makedirs(result_folder, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor()
    ])

    val_dataset = ObjectRemovalDataset(data_dir=opt.data_dir, data_type="val", transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=os.cpu_count())

    onnx_model_path = os.path.join("test_models", f"ia_gen_{opt.chkpt_no}.onnx")
    session = ort.InferenceSession(onnx_model_path)

    for batch_idx, (real_images, masks, ground_truths) in enumerate(val_loader):
        if batch_idx >= opt.num_batches:  # Stop after processing `num_batches`
            break

        st = time()
        
        # Convert inputs to numpy
        img_np = real_images.numpy().astype(np.float32)
        mask_np = masks.numpy().astype(np.float32)

        inputs = {"input_image": img_np, "input_mask": mask_np}
        outputs = session.run(None, inputs)
        coarse_outputs, refined_outputs = torch.tensor(outputs[0]), torch.tensor(outputs[1])

        # Create infused outputs
        coarse_infused = real_images * (1 - masks) + coarse_outputs * masks
        refined_infused = real_images * (1 - masks) + refined_outputs * masks

        for i in range(min(5, real_images.shape[0])):  # Save only first 5 images per batch
            input_img = tensor_to_image(real_images[i])
            mask_img = np.clip(masks[i].cpu().numpy().squeeze() * 255, 0, 255).astype(np.uint8)
            coarse_img = tensor_to_image(coarse_outputs[i])
            refined_img = tensor_to_image(refined_outputs[i])
            coarse_infused_img = tensor_to_image(coarse_infused[i])
            refined_infused_img = tensor_to_image(refined_infused[i])

            img_id = f"batch{batch_idx}_sample{i}"
            Image.fromarray(input_img).save(os.path.join(result_folder, f"{img_id}_input.jpg"))
            Image.fromarray(mask_img).save(os.path.join(result_folder, f"{img_id}_mask.jpg"))
            Image.fromarray(coarse_img).save(os.path.join(result_folder, f"{img_id}_coarse.jpg"))
            Image.fromarray(refined_img).save(os.path.join(result_folder, f"{img_id}_refined.jpg"))
            Image.fromarray(coarse_infused_img).save(os.path.join(result_folder, f"{img_id}_coarse_infused.jpg"))
            Image.fromarray(refined_infused_img).save(os.path.join(result_folder, f"{img_id}_refined_infused.jpg"))

        et = time()
        print(f"Batch {batch_idx + 1} processed in {(et - st):.2f} s. Results in {result_folder}")