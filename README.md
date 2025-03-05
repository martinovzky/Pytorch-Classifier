# Indoor Scene Classification with EfficientNet-B0 
## Overview
This project fine-tunes **EfficientNet-B0** on the MIT Indoor Scenes dataset to classify images into **67** different indoor categories.


## Dataset
- **MIT Indoor Scenes Dataset**: A collection of **15,620** images spanning **67** indoor categories.
- Dataset source: [MIT Indoor Scenes](https://paperswithcode.com/dataset/mit-indoors-scenes)
- Data preprocessing:
  - **Resized** images to match EfficientNet-B0's expected input.
  - **Normalized** pixel values to match ImageNet statistics.
  - **Data augmentation** (random flips, rotations) applied for generalization.


## Model & Training Setup
- **Pretrained Model**: EfficientNet-B0 (pretrained on ImageNet)
- **Fine-Tuning Strategy**: Unfreeze last layers & retrain on MIT Indoor Scenes
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam
- **Batch Size**: *32*
- **Learning Rate**: *1e-3*
- **Epochs**: *10*
- **Validation Split**: *80/20*
- **Hardware**: Google Colab (GPU acceleration enabled)

## Installation & Usage

### **Run in Google Colab (Recommended)**
1. Open `Pytorch_Classifier.ipynb` in **Google Colab**.
2. Enable **GPU**:  
   - Go to `Runtime > Change runtime type > GPU`
3. Follow the notebook instructions to train and validate the model.

### **Run Locally**
1. **Clone the repository**:
   ```bash
   git clone https://github.com/martinovzky/Pytorch-Classifier.git
   cd Pytorch-Classifier
   ```



