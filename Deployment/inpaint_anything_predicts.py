import torch
import numpy as np
import onnxruntime as ort
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from utils import tensor_to_image

def make_inference(image_bytes, mask_bytes, onnx_model_path):
    # Load image and mask from in-memory bytes.
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    image = transform(Image.open(BytesIO(image_bytes)).convert("RGB")).unsqueeze(0)  # (1,3,512,512)
    mask = transform(Image.open(BytesIO(mask_bytes)).convert("L")).unsqueeze(0)      # (1,1,512,512)
    
    image, mask = image.float(), mask.float()
    session = ort.InferenceSession(onnx_model_path)

    img_np_orig = image.numpy().astype(np.float32) 
    img_np = image.numpy().astype(np.float32)
    mask_np = mask.numpy().astype(np.float32)

    step_size = 512
    num_steps = 512 // step_size

    if num_steps > 1:
        for i in range(num_steps):
            for j in range(num_steps):
                x_start, x_end = i * step_size, (i + 1) * step_size
                y_start, y_end = j * step_size, (j + 1) * step_size

                # Extract partial mask for current patch.
                partial_mask = np.zeros_like(mask_np)
                partial_mask[:, :, x_start:x_end, y_start:y_end] = mask_np[:, :, x_start:x_end, y_start:y_end]

                if np.sum(partial_mask) == 0:
                    continue

                inputs = {"input_image": img_np, "input_mask": partial_mask}
                outputs = session.run(["coarse_image", "refined_image"], inputs)
                refined_output = torch.tensor(outputs[1])

                # Update the image in the masked region.
                img_np[:, :, x_start:x_end, y_start:y_end] = refined_output[:, :, x_start:x_end, y_start:y_end]
                img_np[0] = img_np[0] * mask_np[0] + img_np_orig[0] * (1 - mask_np[0])
    
    # Run inference on the full image.
    inputs = {"input_image": img_np, "input_mask": mask_np}
    outputs = session.run(["coarse_image", "refined_image"], inputs)
    coarse_output, refined_output = torch.tensor(outputs[0]), torch.tensor(outputs[1])

    # Blend the outputs with the original image using the mask.
    coarse_in_output = coarse_output * torch.tensor(mask_np) + torch.tensor(img_np_orig) * (1 - torch.tensor(mask_np))
    refined_in_output = refined_output * torch.tensor(mask_np) + torch.tensor(img_np_orig) * (1 - torch.tensor(mask_np))

    # Save final outputs to an in-memory dictionary.
    results_dict = {}
    buf = BytesIO()

    Image.fromarray(tensor_to_image(torch.tensor(img_np_orig[0]))).save(buf, format="JPEG")
    results_dict["original_image.jpg"] = buf.getvalue()
    buf.seek(0); buf.truncate(0)

    Image.fromarray((mask_np[0, 0] * 255).astype(np.uint8)).save(buf, format="JPEG")
    results_dict["original_mask.jpg"] = buf.getvalue()
    buf.seek(0); buf.truncate(0)

    Image.fromarray(tensor_to_image(coarse_in_output[0])).save(buf, format="JPEG")
    results_dict["coarse_in_output.jpg"] = buf.getvalue()
    buf.seek(0); buf.truncate(0)

    Image.fromarray(tensor_to_image(refined_in_output[0])).save(buf, format="JPEG")
    results_dict["refined_in_output.jpg"] = buf.getvalue()
    buf.seek(0); buf.truncate(0)

    Image.fromarray(tensor_to_image(coarse_output[0])).save(buf, format="JPEG")
    results_dict["coarse_output.jpg"] = buf.getvalue()
    buf.seek(0); buf.truncate(0)

    Image.fromarray(tensor_to_image(refined_output[0])).save(buf, format="JPEG")
    results_dict["refined_output.jpg"] = buf.getvalue()
    buf.close()

    return results_dict
