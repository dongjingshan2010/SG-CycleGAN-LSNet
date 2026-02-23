# SG-CycleGAN-LSNet
Code for SG-CycleGAN &amp; LSNet: A two-stage framework for endometrial carcinoma screening. Features SG-CycleGAN for structure-preserving MRI-to-ultrasound synthesis to overcome data scarcity, and LSNet, a lightweight classifier using gradient distillation for efficient and accurate myometrial invasion detection.

# SG-CycleGAN & LSNet for Endometrial Carcinoma Screening

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

PyTorch implementation of the paper: "Lightweight Model with Gradient Distillation for Endometrial Carcinoma Screening: Trained on Augmented MRI-to-Ultrasound Images Synthesized by Structure-Guided Cycle-Consistent Adversarial Network".

📖 Overview

This repository provides an end-to-end deep learning framework designed to address the challenges of data scarcity, class imbalance, and subtle imaging features in ultrasound-based Endometrial Carcinoma (EC) screening. The framework consists of two main stages:

1. SG-CycleGAN (Structure-Guided Cycle-Consistent Adversarial Network): Synthesizes diverse, high-fidelity ultrasound images from unpaired MRI data. It utilizes a Modality-Agnostic Feature Extractor (MAFE) and feature consistency loss to explicitly preserve critical anatomical structures (e.g., the endometrium-myometrium junction) during cross-modal translation.

2. LSNet (Lightweight Screening Network): A computationally efficient classifier built upon MobileViT. It employs a novel Gradient Distillation mechanism to transfer discriminative knowledge from a larger teacher model, guiding a sparse attention mechanism to focus strictly on pathologically relevant regions (like invasion interfaces) while maintaining a very low FLOP count (0.289 GFLOPs).

⚙️ Requirements

Ensure you have Python 3.8+ installed. The required dependencies are listed below:

```bash
pip install torch torchvision torchaudio
pip install scikit-learn scikit-image pandas seaborn matplotlib
pip install einops pyyaml opencv-python
pip install thop fvcore warmup_scheduler

```

---

📂 Dataset Preparation

1. SG-CycleGAN Dataset

Organize your unpaired MRI and Ultrasound (Uto) images into the following directory structure for cross-modal synthesis:

```text
datasets/Med_shallowdeep/
├── trainA/      # MRI images
├── trainB/      # Ultrasound images
├── testA/       # MRI test images
└── testB/       # Ultrasound test images

```

2. LSNet Classification Dataset

The classification network uses a hybrid dataset (real + synthetic images) for pre-training, and real images for fine-tuning.

```text
data_split/
├── synthetic_train/   # Hybrid dataset (Real Ultrasound + SG-CycleGAN Synthetic Ultrasound)
│   ├── Normal/
│   └── Abnormal/      # Superficial & Deep Invasion
├── train/             # Real Ultrasound training data only
├── val/               # Validation data
└── test/              # Test data

```

(Note: The `medical_image_loader.py` also supports multi-class loading for Normal, Superficial, and Deep categories if needed).

---

🚀 Training & Evaluation

Stage 1: Train SG-CycleGAN

Run the following command to train the SG-CycleGAN model. The script supports Distributed Data Parallel (DDP) for multi-GPU training.

```bash
python SG-CycleGAN.py \
  --dataroot ./datasets/Med_shallowdeep/ \
  --epoch 0 \
  --n_epochs 50 \
  --batchSize 32 \
  --input_nc 3 \
  --output_nc 3 \
  --cuda

```

*Note: The code also contains an experimental Rectified Flow generation stage. By default, if `epoch == n_epochs`, the script bypasses Rectified Flow and relies purely on the robust GAN-based generation.*

Stage 2: Train and Evaluate LSNet

The `LSNet.py` script automatically handles the two-stage training protocol (pre-training on the hybrid dataset followed by fine-tuning on the real dataset) and executes three independent runs with different random seeds for rigorous statistical validation.

```bash
python LSNet.py \
  --bs 32 \
  --n_epochs 50 \
  --lr 5e-4 \
  --imagesize 320 \
  --num_classes 2 \
  --opt adam

```

What the script does automatically:

1. Computes FLOPs and parameter counts using `thop` and `fvcore`.
2. Pre-trains the student and teacher models concurrently using gradient distillation.
3. Fine-tunes the model on real ultrasound data.
4. Evaluates on the held-out test sets (`test/`, etc.) and calculates comprehensive metrics (Accuracy, Sensitivity, Specificity, Precision, F1-score, ROC-AUC).
5. Generates performance plots and saves detailed CSV reports in the `./results/` folder.

---

📊 Pre-trained Models

(Optional: Provide links to download your pre-trained weights for `improved-net.pth` and `shared_generator.pth` here if you plan to open-source them).

---

📝 Citation

If you find this code or our paper useful in your research, please consider citing:

```bibtex
@article{shan2026lightweight,
  title={Efficient endometrial carcinoma screening via cross-modal synthesis and gradient distillation},
  author={Shan, Dongjing and Luo, Yamei and Xuan, Jiqing and Huang, Lu and Li, Jin and Yang, Mengchu and Chen, Zeyu and Lv, Fajin and Tang, Yong and Zhang, Chunxiang},
  journal={arXiv},
  year={2026}
}



