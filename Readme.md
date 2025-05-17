# PlantStressAI

## Overview
PlantStressAI is a research project focused on developing lightweight AI models for diagnosing plant stress in small-scale farming. The project compares Vision Transformers (ViTs) and Convolutional Neural Networks (CNNs) to identify environmental and nutritional stresses in crops, optimizing for edge deployment on resource-constrained devices like smartphones or drones. The primary dataset used is EarlyNSD, with additional experiments incorporating the Groundnut Leaf Dataset.

This repository contains the code, datasets, and documentation for the project conducted by Janarthan Aravindan Aathavan, Ellawela Liyanage Jayathilake from the University at Buffalo.

## Project Goals
- Develop and compare lightweight AI models (ViT-Tiny, ResNet-18, MobileViT, and ViT with spatial attention) for plant stress classification.
- Optimize models for low-power edge devices to support smallholder farmers.
- Address issues like limited dataset size and class imbalance through techniques such as focal loss and Masked Autoencoder (MAE) pre-training.
- Evaluate model performance using accuracy, F1-score, Intersection over Union (IoU), and inference time.

## Setup
- **Python**: 3.10.11
- **Install dependencies**:
  ```bash
  pip install -r requirements.txt
  ```

## Datasets
- **EarlyNSD Dataset**:
  - Source: Kaggle (see report for link)
  - Size: 2,700 RGB images of cucurbit leaves (ash gourd, bitter gourd, snake gourd)
  - Classes: 9 (fresh leaves, nitrogen deficiency, potassium deficiency for each crop)
  - Split: 1,890 training, 405 validation, 405 testing
  - Challenges: Varying lighting, background noise, subtle symptom differences
  - Directory structure:
    ```
    NSD/
    ├── train/
    │   ├── class1/
    │   └── ...
    ├── val/
    │   ├── class1/
    │   └── ...
    ├── test/
    │   ├── class1/
    │   └── ...
    ```
- **Groundnut Leaf Dataset**:
  - Source: Kaggle
  - Includes stresses like nitrogen deficiency, phosphorus deficiency, overwatering, and drought
  - Used for additional training and model generalization

## Models
The project evaluates four main models:
1. **ViT-Tiny with MAE**:
   - Lightweight Vision Transformer with 5.7M parameters
   - Pre-trained with Masked Autoencoder (MAE) for robust feature extraction
   - Fine-tuned on EarlyNSD with data augmentation
2. **ResNet-18**:
   - Baseline CNN with residual connections
   - Used as a benchmark for comparing CNNs with Vision Transformers
3. **MobileViT**:
   - Hybrid model combining ViT and CNN features
   - Variants: MobileViT-S, MobileViT-XS, MobileViT-XXS
   - Designed for efficient deployment on edge devices
4. **ViT with Spatial Attention (Twins-SVT-S)**:
   - Incorporates local and global attention mechanisms
   - Focuses on important image regions like leaf veins and discoloration patterns

## Training
- **Hyperparameters**:
  - Learning rate: 1e-5
  - Epochs: 5–20 (varies by experiment)
  - Input resolution: 224x224
  - Data augmentation: Color jittering, random cropping, Gaussian noise
- **Focal Loss**: Used to address class imbalance, improving minority class separation
- **Environment**: Google Colab Pro with Tesla T4 GPU

## Instructions
- Ensure the NSD dataset is placed in the `NSD/` directory with the structure shown above.
- Open the `expt_1.ipynb` notebook and run all cells one by one.
- Then, run the `expt_focal.ipynb` notebook, which includes the addition of focal loss to boost performance metrics.

## Notes
- GPU recommended; CPU supported as well.

## Acknowledgments
We thank the University at Buffalo for support and the contributors of the EarlyNSD and Groundnut Leaf Datasets on Kaggle.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
