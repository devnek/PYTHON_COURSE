# PyTorch: Zero to Production 🔥

A comprehensive, hands-on PyTorch course from absolute basics to building and deploying a real AI model — packed with **corner cases, pitfalls, and best practices** that tutorials usually skip.

---

## 📖 Course Overview

10 chapters covering tensors, autograd, data pipelines, neural networks, training loops, deployment, and a complete end-to-end AI project.

### 🟢 Foundations (Chapters 1–4)
- **Chapter 1:** [Introduction & Setup](Ch01_Introduction_and_Setup.ipynb) — What is PyTorch, installation (CPU/GPU/MPS), device management, and GPU memory corner cases.
- **Chapter 2:** [Tensors Deep Dive](Ch02_Tensors_Deep_Dive.ipynb) — Creating, indexing, reshaping tensors. ⚠️ Corner cases: `view()` vs `reshape()`, contiguous memory, views sharing memory, dtype overflow.
- **Chapter 3:** [Tensor Operations & Math](Ch03_Tensor_Operations_and_Math.ipynb) — Element-wise ops, matrix multiplication, broadcasting rules, reductions. ⚠️ Corner cases: silent broadcasting bugs, `keepdim`, in-place ops with autograd.
- **Chapter 4:** [Autograd & Gradients](Ch04_Autograd_and_Gradients.ipynb) — Computation graphs, `.backward()`, gradient accumulation. ⚠️ Corner cases: forgetting `zero_grad()`, non-scalar backward, `detach()` sharing memory.

### 🟡 Building Blocks (Chapters 5–7)
- **Chapter 5:** [Datasets & DataLoaders](Ch05_Datasets_and_DataLoaders.ipynb) — Custom datasets, transforms, batching, workers. ⚠️ Corner cases: `num_workers` on macOS, reproducibility, imbalanced data with `WeightedRandomSampler`.
- **Chapter 6:** [Building Neural Networks](Ch06_Building_Neural_Networks.ipynb) — `nn.Module`, `nn.Sequential`, common layers (Linear, Conv2d, LSTM), activations. ⚠️ Corner cases: never call `forward()` directly, weight initialization, dying ReLU, softmax `dim=`.
- **Chapter 7:** [Training Loop Mastery](Ch07_Training_Loop_Mastery.ipynb) — Loss functions, optimizers, the complete train/val loop. ⚠️ Corner cases: `train()`/`eval()` toggle, Adam vs AdamW, gradient clipping, device ordering.

### 🔴 Advanced & Production (Chapters 8–10)
- **Chapter 8:** [Saving, Loading & Deployment](Ch08_Saving_Loading_and_Deployment.ipynb) — `state_dict`, checkpoints, TorchScript. ⚠️ Corner cases: `map_location`, `strict=False`, architecture mismatches.
- **Chapter 9:** [Advanced Techniques & Pitfalls](Ch09_Advanced_Techniques_and_Corner_Cases.ipynb) — LR schedulers, mixed precision (AMP), gradient accumulation, layer freezing. Includes the **complete 15-point pitfall checklist**.
- **Chapter 10:** [🎯 AI Project: MNIST Digit Classifier](Ch10_AI_Project_Image_Classifier.ipynb) — End-to-end CNN project: data augmentation → training with early stopping → confusion matrix evaluation → single-image inference → TorchScript export.

---

## 🚀 Getting Started

```bash
# Install PyTorch (CPU)
pip install torch torchvision torchaudio

# Open notebooks
jupyter notebook
```

## ⚠️ Corner Cases Covered

This course highlights **15+ real-world pitfalls** that cause silent bugs, crashes, or poor performance — the kind of issues you only discover after hours of debugging. Each is marked with ⚠️ throughout the notebooks.
