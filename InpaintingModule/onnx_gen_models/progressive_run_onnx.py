import os
import torch
import numpy as np
import onnxruntime as ort
import torchvision.transforms as transforms
from PIL import Image

def tensor_to_image(tensor):
    np_img = tensor.cpu().numpy()
    np_img = np.clip(np_img * 255, 0, 255).astype(np.uint8)
    return np.transpose(np_img, (1, 2, 0))

def process_image(image_path, mask_path, onnx_model_path, save_folder):
    # Load image & mask
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)  # (1,3,512,512)
    mask = transform(Image.open(mask_path).convert("L")).unsqueeze(0)      # (1,1,512,512)
    
    image, mask = image.float(), mask.float()
    session = ort.InferenceSession(onnx_model_path)

    os.makedirs(save_folder, exist_ok=True)

    img_np_orig = image.numpy().astype(np.float32) 
    img_np = image.numpy().astype(np.float32)
    mask_np = mask.numpy().astype(np.float32)
    mask_np = 1 - mask_np

    step_size = 512
    num_steps = 512 // step_size

    for i in range(num_steps):
        for j in range(num_steps):
            x_start, x_end = i * step_size, (i + 1) * step_size
            y_start, y_end = j * step_size, (j + 1) * step_size

            # Extract current 64x64 mask
            partial_mask = np.zeros_like(mask_np)
            partial_mask[:, :, x_start:x_end, y_start:y_end] = mask_np[:, :, x_start:x_end, y_start:y_end]

            if np.sum(partial_mask) == 0:
                continue

            # Run inference
            inputs = {"input_image": img_np, "input_mask": partial_mask}
            outputs = session.run(None, inputs)
            refined_output = torch.tensor(outputs[0])  # Ensure correct output index

            # Update the image in the masked region
            img_np[:, :, x_start:x_end, y_start:y_end] = refined_output[:, :, x_start:x_end, y_start:y_end]
            img_np[0] = img_np[0] * mask_np[0] + img_np_orig[0] * (1 - mask_np[0])
            
            # Save intermediate results (optional)
            iter_id = f"iter_{i}_{j}"
            Image.fromarray(tensor_to_image(torch.tensor(img_np[0]))).save(os.path.join(save_folder, f"{iter_id}_img.jpg"))
            Image.fromarray((partial_mask[0, 0] * 255).astype(np.uint8)).save(os.path.join(save_folder, f"{iter_id}_mask.jpg"))

    # Run inference
    inputs = {"input_image": img_np, "input_mask": mask_np}
    outputs = session.run(None, inputs)
    refined_output = torch.tensor(outputs[0])  # Ensure correct output index

    # Update the image in the masked region
    img_np[:, :, x_start:x_end, y_start:y_end] = refined_output[:, :, x_start:x_end, y_start:y_end]
    img_np[0] = img_np[0] * mask_np[0] + img_np_orig[0] * (1 - mask_np[0])
    Image.fromarray(tensor_to_image(torch.tensor(img_np[0]))).save(os.path.join(save_folder, "final_result.jpg"))


if __name__ == "__main__":
    image_path = "data/val/img/I-210618_I01001_W01/I-210618_I01001_W01_F0042.jpg"
    mask_path = "data/val/mask/I-210618_I01001_W01/I-210618_I01001_W01_F0042_M.png"
    onnx_model_path = "test_models/ia_gen_55.onnx"
    save_folder = "test_models/Results"
    
    process_image(image_path, mask_path, onnx_model_path, save_folder)
