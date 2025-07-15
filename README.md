# DeepRazor: AI-Powered Object Removal Tool
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-green.svg)](https://deeprazor.streamlit.app/)
[![Demo Video](https://img.shields.io/badge/Demo-Watch%20Video-red.svg)](https://www.youtube.com/watch?v=IRND7na0cWA)

**Remove unwanted objects from photos with AI - No coding required!**

DeepRazor makes professional photo editing accessible to everyone. Simply upload your image, mark what you want to remove, and let our AI seamlessly fill in the background. Perfect for social media, real estate, e-commerce, or just cleaning up your personal photos.

---

## 🎯 For Users: Get Started in 2 Minutes

### What DeepRazor Can Do
- **🎨 Remove Any Object**: People, cars, text, unwanted items - anything!
- **🖌️ Smart Background Fill**: AI automatically reconstructs what was behind the object
- **🚀 Multiple Ways to Select**: Draw masks manually or let AI detect objects automatically
- **💻 Web Interface**: No coding needed - just point, click, and download
- **⚡ Fast Results**: Get professional results in seconds

### Quick Start (Zero Setup Required)
```bash

# 1. Clone and install
git clone https://github.com/ShubhamKNIT/DeepRazor.git
cd DeepRazor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Only for Linux/Ubuntu 
sudo apt-get update
sudo apt-get install freeglut3-dev libgtk2.0-dev

# 2. Launch the app
cd Deployment
streamlit run app.py

# 3. Open http://localhost:8501 in your browser and start editing!
```

### 🎭 Four Easy-to-Use Tools

| Tool | What It Does | Best For |
|------|--------------|----------|
| **🎓 How to Use** | Interactive tutorial with demo video | First-time users |
| **✏️ Draw Mask** | Manually draw what to remove | Precise selection, complex shapes |
| **🎯 YOLO Detection** | AI detects mask of 80+ object types automatically | Common objects (people, cars, animals) |
| **🎨 Inpaint Anything** | Upload image + mask for instant results | Quick masked object removal |

### Supported File Types
- **Input**: JPG, JPEG, PNG images
- **Output**: High-quality PNG images
- **Mask**: PNG images (black = keep, white = remove)

---

## 💼 For Developers: Integration & Customization

### Python API Usage
```python
from inpaint_anything_predicts import make_inference

# Basic usage
image_bytes = open("photo.jpg", "rb").read()
mask_bytes = open("mask.png", "rb").read()
result = make_inference(image_bytes, mask_bytes, "onnx_gen_models/ia_gen_39.onnx")

# Save result
with open("result.png", "wb") as f:
    f.write(result)
```

### Project Structure
```
DeepRazor/
├── Deployment/                         # 🌐 Web Application
│   ├── app.py                         # Main Streamlit app
│   ├── how_to_use.py                  # Tutorial page
│   ├── draw_mask.py                   # Manual mask drawing
│   ├── segment_anything.py            # YOLO object detection
│   ├── inpaint_anything.py            # Core inpainting interface
│   └── onnx_gen_models/               # Pre-trained models
│       ├── ia_gen_10.onnx ... ia_gen_59.onnx
│       └── run_onnx.py                # ONNX inference
├── InpaintingModule/                   # 🔬 Research & Training
│   ├── train.py                       # Training entry point
│   ├── test.py                        # Testing entry point
│   ├── validate.py                    # Validation entry point
│   ├── src/dataset/                   # Dataset preparation scripts
│   ├── src/model/                     # Model architectures
│   ├── src/utils/                     # Training utilities
│   └── Logs and Results               # Training Logs and Results
└── requirements.txt                    # Dependencies
```

---

## 🔬 For Researchers: Technical Architecture & Training

### Model Architecture Overview
DeepRazor implements a **two-stage Generative Adversarial Network** using **Contextual Residual Aggregation (CRA)**:

```
Input Image + Mask
       ↓
Coarse Generator (256×256) → Coarse Inpainting
       ↓
Refinement Generator (512×512) → Final Result
       ↓
Discriminator → Quality Assessment
```

### Key Technical Innovations

#### 1. Two-Stage Generator Design
- **Coarse Generator**: Fast 256×256 initial inpainting with gated convolutions
- **Refinement Generator**: High-quality 512×512 enhancement with contextual attention
- **Progressive Training**: Coarse stage first, then joint refinement training

#### 2. Advanced Components
- **Gated Convolutions**: Dynamic feature selection based on mask validity
- **Contextual Attention Module (CAM)**: Transfers context from valid to masked regions
- **Multi-Scale Dilated Blocks**: Capture context at rates [1, 2, 4, 8, 16]
- **Swin Transformer Integration**: Global context modeling in final training epochs (experimental)

#### 3. Comprehensive Loss Design
```python
# Coarse Stage
L_coarse = λ_L1 * L1_loss + λ_FFT * Fourier_loss + λ_TV * TV_loss

# Refinement Stage  
L_refine = λ_L1 * L1_loss + λ_FFT * Fourier_loss + λ_adv * Adversarial_loss + 
           λ_perceptual * VGG_loss + λ_FM * Feature_matching_loss

# Discriminator
L_disc = Hinge_loss + Gradient_penalty
```

### Dataset Information (RORD)
- **Source**: Real-world Object Removal Dataset
- **Size**: 236GB original → 41.5GB prepared subset
- **Training**: ~24K high-quality triplets [Image, Mask, Ground Truth]
- **Validation**: ~4K instances for evaluation
- **Resolution**: 512×512 for training, supports higher resolutions

### Training Your Own Models

#### DeepRazor Module Documentation

Detailed explanation of all core modules, functions, and architectural breakdown.

📄 [Read the full Module Documentation (PDF)](DeepRazorDocs.pdf)


#### Setup Training Environment
```bash
# 1. Prepare dataset
cd RORD-dataset-preparation
python prepare_dataset.py  # Follow dataset preparation guide

# 2. Configure training
cd InpaintingModule
vim env_var.py  # Set dataset paths and hyperparameters

# 3. Start training
python train.py --start_epoch 1 --num_epochs 50 --batch_size 16 --device cuda
```

#### Advanced Training Features
- **Modular Design**: Easily extend or replace components
- **Mixed Precision Training**: 50% memory reduction with AMP
- **Resume Training**: Automatic checkpoint loading
- **Real-time Monitoring**: Live loss tracking and GPU usage
- **Validation Metrics**: PSNR, SSIM, FID, LPIPS evaluation
- **Result Visualization**: Sample outputs saved during training

#### Export to ONNX
```bash
cd Deployment/onnx_gen_models
python export_to_onnx.py --chkpt_no 50 --model_path ../../InpaintingModule/checkpoints/
```

### Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **PSNR** | 21.5+ dB | Peak Signal-to-Noise Ratio |
| **SSIM** | 0.86+ | Structural Similarity Index |
| **FID** | 20-28 | Fréchet Inception Distance |
| **LPIPS** | 0.25-0.26 | Learned Perceptual Image Patch Similarity |

### Research Applications
- **Image Inpainting**: Core inpainting research and benchmarking
- **Object Removal**: Real-world object removal in natural images
- **Context Understanding**: Studying contextual attention mechanisms
- **GAN Training**: Advanced adversarial training techniques
- **Loss Function Design**: Multi-objective optimization in computer vision

---

## 🤝 Contributing & Community

### Ways to Contribute
- 🐛 **Bug Reports**: Found an issue? Create a detailed bug report
- 💡 **Feature Requests**: Suggest new functionality or improvements  
- 🔧 **Code Contributions**: Submit pull requests for bug fixes or features
- 📖 **Documentation**: Improve tutorials, examples, or API docs
- 🎓 **Research**: Share new architectures, loss functions, or training techniques

### Development Workflow
```bash
# 1. Fork and clone
git fork https://github.com/ShubhamKNIT/DeepRazor.git
git clone https://github.com/YourUsername/DeepRazor.git
cd DeepRazor

# 2. Create feature branch
git checkout -b feature/amazing-improvement

# 3. Make changes and test
pip install -r requirements.txt
# Test your changes thoroughly

# 4. Submit pull request
git commit -am "Add amazing improvement"
git push origin feature/amazing-improvement
# Create PR on GitHub
```
---

## 📚 References & Acknowledgments

### Core Research
1. **Sagong, M.-C., et al. (2022)**  
   *RORD: A Real-world Object Removal Dataset*  
   BMVC 2022 - [Paper](https://bmvc2022.mpi-inf.mpg.de/0542.pdf)

2. **Yi, Z., et al. (2020)**  
   *Contextual Residual Aggregation for Ultra High-Resolution Image Inpainting*  
   arXiv:2005.09704 - [Paper](https://arxiv.org/pdf/2005.09704)

3. **Yu, J., et al. (2019)**  
   *Free-form Image Inpainting with Gated Convolution*  
   ICCV 2019 - [Paper](https://arxiv.org/pdf/1806.03589)

### Special Thanks
- **RORD Dataset** creators for high-quality training data
- **Ultralytics YOLO** team for object detection capabilities  
- **Streamlit** community for the amazing web framework
- **PyTorch** and **ONNX** teams for ML infrastructure
- **FastSAM** contributors for rapid segmentation
- **Open Source Community** for inspiration and support

---

**🎨 Turn any photo into a masterpiece - Try DeepRazor today!**

*Made with ❤️ by [Shubham Kumar](https://github.com/ShubhamKNIT) | Star ⭐ if you find this useful!*
