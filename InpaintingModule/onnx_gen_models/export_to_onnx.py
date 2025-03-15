import sys
sys.path.append("/home/jupyter/InpaintingModule")

import torch
from src.model.generator import *
import torch.onnx
import argparse


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--chkpt_no', type = int, help = 'enter a checkpoint no to export to onnx model format')
    opt = parser.parse_args()
    chkpt_no = opt.chkpt_no
    
    chkpt_path = f'Checkpoints/checkpoint_{chkpt_no}.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator()
    generator.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(chkpt_path, map_location=device)
    generator.load_state_dict(checkpoint["gen_state_dict"], strict=False)
    generator.eval()
    
    # Create dummy inputs on the proper device
    img = torch.randn(1, 3, 512, 512, device=device)
    mask = torch.randn(1, 1, 512, 512, device=device)
    
    # inputs[img, mask], outputs[coarse, refined]
    input_names = ["input_image", "input_mask"]
    output_names = ["coarse_image", "refined_image"]
    dynamic_axes = {
        "input_image": {0: "batch_size"},
        "input_mask": {0: "batch_size"},
        "coarse_image": {0: "batch_size"},
        "refined_image": {0: "batch_size"}
    }
    
    torch.onnx.export(
        generator,
        (img, mask),
        f'ia_gen_{chkpt_no}.onnx',
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=True,
        opset_version=11
    )
