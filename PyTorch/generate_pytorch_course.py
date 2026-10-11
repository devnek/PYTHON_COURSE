#!/usr/bin/env python3
"""
Generate a comprehensive PyTorch course as Jupyter Notebooks.
10 chapters: foundations → intermediate → corner cases → real AI project.
"""

import json, os

OUTDIR = "/Users/devanayak/Documents/PROJECTS/PYTHON_AI/FUNDAMENTALS/python_course/PyTorch"


def make_nb(cells_data):
    cells = []
    for ctype, src in cells_data:
        lines = src.split("\n")
        source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": ctype, "metadata": {}, "source": source}
        if ctype == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def save(filename, cells_data):
    path = os.path.join(OUTDIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_nb(cells_data), f, indent=1)
    print(f"  ✅ {path}")


# ═════════════════════════════════════════════════════════════
# CHAPTER 1 — INTRODUCTION & SETUP
# ═════════════════════════════════════════════════════════════

def ch01():
    save("Ch01_Introduction_and_Setup.ipynb", [
        ("markdown", """# 🔥 Chapter 1 — Introduction to PyTorch & Environment Setup

---

### 🎯 What you'll learn
- What PyTorch is and why it dominates AI research
- PyTorch vs TensorFlow: key differences
- Installing PyTorch (CPU & GPU)
- Verifying your installation
- Understanding CUDA and device management"""),

        ("markdown", """## 1.1 — What is PyTorch?

PyTorch is an open-source deep learning framework developed by **Meta AI (Facebook)**. It provides:

- **Dynamic Computation Graphs** (define-by-run): Build graphs on-the-fly, making debugging natural with standard Python tools.
- **Pythonic API**: Feels like writing NumPy, but with GPU acceleration and automatic differentiation.
- **Production Ready**: `TorchScript` and `torch.compile()` for deployment; ONNX export for cross-platform inference.

### PyTorch vs TensorFlow

| Feature | PyTorch | TensorFlow |
|---|---|---|
| Graph Type | Dynamic (eager by default) | Static (eager optional in TF2) |
| Debugging | Standard Python debugger | Harder to debug graph mode |
| Research Adoption | ~80% of papers (2024) | Declining in research |
| Production | TorchServe, torch.compile | TF Serving, TF Lite |
| Learning Curve | Gentle, Pythonic | Steeper, more boilerplate |"""),

        ("markdown", """## 1.2 — Installation

### CPU Only
```bash
pip install torch torchvision torchaudio
```

### With CUDA (NVIDIA GPU)
```bash
# Check your CUDA version first: nvidia-smi
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### macOS Apple Silicon (MPS backend)
```bash
pip install torch torchvision torchaudio
# MPS backend is automatically available on M1/M2/M3 chips
```"""),

        ("markdown", "## 1.3 — Verifying Your Installation"),

        ("code", """import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version:    {torch.version.cuda}")
    print(f"GPU Device:      {torch.cuda.get_device_name(0)}")
    print(f"GPU Count:       {torch.cuda.device_count()}")

# macOS Apple Silicon
print(f"MPS Available:   {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else 'N/A'}")"""),

        ("markdown", """## 1.4 — Device Management (CPU / GPU / MPS)

> ⚠️ **Corner Case**: Always write device-agnostic code. Never hardcode `"cuda"`. Use the pattern below."""),

        ("code", """import torch

# ✅ Best practice: device-agnostic selection
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

# Moving tensors to device
x = torch.randn(3, 3)
x = x.to(device)
print(f"Tensor device: {x.device}")"""),

        ("markdown", """## ⚠️ Corner Case: GPU Memory Management"""),

        ("code", """# Check GPU memory (if CUDA available)
if torch.cuda.is_available():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print(f"Cached:    {torch.cuda.memory_reserved() / 1e6:.1f} MB")
    
    # Free cached memory
    torch.cuda.empty_cache()
    
    # ⚠️ Corner Case: memory_reserved ≠ memory_allocated
    # PyTorch uses a caching allocator — reserved > allocated is normal.
    # Only worry if allocated keeps growing (memory leak).
else:
    print("No CUDA device — skip GPU memory checks.")"""),

        ("markdown", "## ✏️ Exercises"),

        ("code", """# Exercise 1: Write a function that prints a full system report:
# Python version, PyTorch version, available device, and (if GPU) GPU name + memory.
# YOUR CODE HERE


# Exercise 2: Create a tensor on CPU, move it to the best available device,
# perform an operation, and move the result back to CPU.
# YOUR CODE HERE
"""),
    ])


# ═════════════════════════════════════════════════════════════
# CHAPTER 2 — TENSORS
# ═════════════════════════════════════════════════════════════

def ch02():
    save("Ch02_Tensors_Deep_Dive.ipynb", [
        ("markdown", """# 🔥 Chapter 2 — Tensors: The Foundation of Everything

---

### 🎯 What you'll learn
- Creating tensors (from data, random, zeros, ones, ranges)
- Tensor attributes: shape, dtype, device
- Indexing, slicing, and advanced indexing
- Reshaping, stacking, concatenating
- ⚠️ Critical corner cases (views vs copies, contiguous memory)"""),

        ("markdown", "## 2.1 — Creating Tensors"),

        ("code", """import torch

# From Python data
t1 = torch.tensor([1, 2, 3])                      # 1D
t2 = torch.tensor([[1, 2], [3, 4]])                # 2D
t3 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 3D

# Special tensors
zeros = torch.zeros(3, 4)         # 3x4 of zeros
ones = torch.ones(2, 3)           # 2x3 of ones
eye = torch.eye(3)                # 3x3 identity matrix
full = torch.full((2, 3), 7.0)    # 2x3 filled with 7.0

# Random tensors
rand = torch.rand(3, 3)           # Uniform [0, 1)
randn = torch.randn(3, 3)         # Normal (mean=0, std=1)
randint = torch.randint(0, 10, (3, 3))  # Random integers [0, 10)

# Sequences
arange = torch.arange(0, 10, 2)          # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)       # [0.0, 0.25, 0.5, 0.75, 1.0]

print("zeros:\\n", zeros)
print("randn:\\n", randn)
print("arange:", arange)
print("linspace:", linspace)"""),

        ("markdown", "## 2.2 — Tensor Attributes"),

        ("code", """t = torch.randn(3, 4, 5)

print(f"Shape:       {t.shape}")        # torch.Size([3, 4, 5])
print(f"Dimensions:  {t.ndim}")          # 3
print(f"Dtype:       {t.dtype}")         # torch.float32
print(f"Device:      {t.device}")        # cpu
print(f"Total elems: {t.numel()}")       # 60 (3*4*5)
print(f"Requires grad: {t.requires_grad}")  # False"""),

        ("markdown", """## 2.3 — ⚠️ Corner Case: Data Types (dtype)

PyTorch defaults to `float32`, but wrong dtypes cause silent bugs or crashes."""),

        ("code", """# ⚠️ Integer tensors lose precision in division!
a = torch.tensor([1, 2, 3])       # int64 by default
print(f"dtype: {a.dtype}")         # torch.int64
print(f"a / 2 = {a / 2}")         # Becomes float64!

# ⚠️ Mixing dtypes
b = torch.tensor([1.0, 2.0, 3.0])  # float32
# a + b works — auto-promotes int64 → float64

# ✅ Best practice: always specify dtype explicitly for ML
x = torch.tensor([1, 2, 3], dtype=torch.float32)
print(f"Explicit dtype: {x.dtype}")

# Type casting
x_int = x.int()           # or x.to(torch.int32)
x_double = x.double()     # or x.to(torch.float64)
x_half = x.half()         # float16 — for mixed precision training

# ⚠️ Corner Case: float16 overflow
big = torch.tensor([65536.0], dtype=torch.float16)
print(f"float16 overflow: {big}")  # inf! float16 max is ~65504"""),

        ("markdown", "## 2.4 — Indexing and Slicing"),

        ("code", """t = torch.arange(20).reshape(4, 5)
print("Original:\\n", t)

# Basic indexing
print(f"Row 0:      {t[0]}")
print(f"Element[1,3]: {t[1, 3]}")
print(f"Last row:   {t[-1]}")

# Slicing
print(f"Rows 1-2:   {t[1:3]}")
print(f"Column 2:   {t[:, 2]}")
print(f"Submatrix:\\n{t[1:3, 2:4]}")

# Boolean indexing (masking)
mask = t > 10
print(f"Elements > 10: {t[mask]}")

# Fancy indexing — select specific indices
rows = torch.tensor([0, 2, 3])
print(f"Rows [0,2,3]:\\n{t[rows]}")"""),

        ("markdown", """## 2.5 — Reshaping Operations

> ⚠️ **Critical Corner Case**: `view()` vs `reshape()` vs `contiguous()`"""),

        ("code", """t = torch.arange(12)

# reshape — always works (may copy if not contiguous)
r = t.reshape(3, 4)
print("Reshaped:\\n", r)

# view — ONLY works on contiguous tensors (no copy, shares memory)
v = t.view(3, 4)

# ⚠️ Corner Case: transpose makes tensors non-contiguous!
t2 = torch.arange(12).reshape(3, 4)
t2_t = t2.t()  # Transpose
print(f"Is transposed contiguous? {t2_t.is_contiguous()}")  # False!

# t2_t.view(12)  # ❌ RuntimeError! Not contiguous
fixed = t2_t.contiguous().view(12)  # ✅ Make contiguous first
# OR just use reshape() — it handles this automatically
fixed2 = t2_t.reshape(12)  # ✅ Always works

print(f"Fixed: {fixed}")"""),

        ("code", """# More reshaping operations
t = torch.randn(2, 3, 4)

# Flatten
flat = t.flatten()                   # → (24,)
flat_partial = t.flatten(1)          # Flatten dims 1+ → (2, 12)
print(f"Flatten all: {flat.shape}, Flatten from dim 1: {flat_partial.shape}")

# Squeeze (remove dimensions of size 1)
t2 = torch.randn(1, 3, 1, 4)
squeezed = t2.squeeze()              # → (3, 4)
squeeze_dim = t2.squeeze(0)          # Only squeeze dim 0 → (3, 1, 4)
print(f"Squeeze all: {squeezed.shape}, Squeeze dim 0: {squeeze_dim.shape}")

# Unsqueeze (add dimension of size 1)
t3 = torch.randn(3, 4)
unsqueezed = t3.unsqueeze(0)         # → (1, 3, 4)  — add batch dim
print(f"Unsqueeze: {unsqueezed.shape}")

# Expand and Repeat
t4 = torch.tensor([[1], [2], [3]])   # (3, 1)
expanded = t4.expand(3, 4)           # → (3, 4) — no memory copy!
repeated = t4.repeat(1, 4)           # → (3, 4) — copies memory
print(f"Expand:\\n{expanded}")
print(f"Repeat:\\n{repeated}")"""),

        ("markdown", "## 2.6 — Stacking and Concatenation"),

        ("code", """a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# cat — join along existing dimension
cat_rows = torch.cat([a, b], dim=0)   # (4, 2) — stack vertically
cat_cols = torch.cat([a, b], dim=1)   # (2, 4) — stack horizontally
print(f"cat dim=0:\\n{cat_rows}")
print(f"cat dim=1:\\n{cat_cols}")

# stack — join along NEW dimension
stacked = torch.stack([a, b], dim=0)  # (2, 2, 2) — creates new dim
print(f"stack:\\n{stacked}")
print(f"stack shape: {stacked.shape}")

# ⚠️ Corner Case: cat vs stack
# cat: tensors must match on all dims EXCEPT the cat dim
# stack: tensors must have IDENTICAL shapes"""),

        ("markdown", """## 2.7 — ⚠️ Corner Case: Views Share Memory!"""),

        ("code", """original = torch.tensor([1, 2, 3, 4, 5])

# view creates a VIEW — shares same underlying data!
view = original.view(1, 5)
view[0, 0] = 99

print(f"Original after modifying view: {original}")
# [99, 2, 3, 4, 5] — CHANGED! They share memory!

# ✅ Use .clone() for an independent copy
original2 = torch.tensor([1, 2, 3, 4, 5])
independent = original2.clone().view(1, 5)
independent[0, 0] = 99
print(f"Original after modifying clone: {original2}")
# [1, 2, 3, 4, 5] — Unchanged!"""),

        ("markdown", "## ✏️ Exercises"),

        ("code", """# Exercise 1: Create a 5x5 identity matrix, then extract the diagonal as a 1D tensor
# YOUR CODE HERE


# Exercise 2: Create a (3, 4) tensor of random integers 0-9.
# Replace all values > 5 with -1 using boolean masking.
# YOUR CODE HERE


# Exercise 3: Create two tensors of shape (2, 3). Stack them to get shape (2, 2, 3),
# then reshape to (4, 3). Verify the result makes sense.
# YOUR CODE HERE
"""),
    ])


# ═════════════════════════════════════════════════════════════
# CHAPTER 3 — TENSOR OPERATIONS & MATH
# ═════════════════════════════════════════════════════════════

def ch03():
    save("Ch03_Tensor_Operations_and_Math.ipynb", [
        ("markdown", """# 🔥 Chapter 3 — Tensor Operations & Mathematical Functions

---

### 🎯 What you'll learn
- Element-wise and matrix operations
- Broadcasting rules and corner cases
- Reduction operations (sum, mean, max, argmax)
- In-place operations and why they're dangerous with autograd"""),

        ("markdown", "## 3.1 — Element-wise Operations"),

        ("code", """import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"Add:      {a + b}")       # or torch.add(a, b)
print(f"Subtract: {a - b}")
print(f"Multiply: {a * b}")       # Element-wise, NOT matrix multiply
print(f"Divide:   {a / b}")
print(f"Power:    {a ** 2}")
print(f"Sqrt:     {torch.sqrt(a)}")
print(f"Abs:      {torch.abs(torch.tensor([-1., 2., -3.]))}")
print(f"Clamp:    {torch.clamp(a, min=1.5, max=2.5)}")  # Clip values"""),

        ("markdown", """## 3.2 — Matrix Operations"""),

        ("code", """A = torch.tensor([[1., 2.], [3., 4.]])
B = torch.tensor([[5., 6.], [7., 8.]])

# Matrix multiplication — 3 equivalent ways
mm = torch.mm(A, B)            # Only for 2D
matmul = torch.matmul(A, B)    # Works for any dims (broadcasting)
at_sign = A @ B                 # Shorthand for matmul

print(f"A @ B:\\n{at_sign}")

# ⚠️ Corner Case: * vs @
print(f"\\nA * B (element-wise):\\n{A * B}")
print(f"A @ B (matrix mult):\\n{A @ B}")
# These are COMPLETELY different! Common bug source.

# Dot product (1D only)
v1 = torch.tensor([1., 2., 3.])
v2 = torch.tensor([4., 5., 6.])
print(f"\\nDot product: {torch.dot(v1, v2)}")

# Batch matrix multiplication
batch_A = torch.randn(10, 3, 4)  # 10 matrices of 3x4
batch_B = torch.randn(10, 4, 5)  # 10 matrices of 4x5
result = torch.bmm(batch_A, batch_B)  # → (10, 3, 5)
print(f"Batch matmul shape: {result.shape}")"""),

        ("markdown", """## 3.3 — ⚠️ Broadcasting Rules

Broadcasting lets tensors of different shapes work together. But it's also a **major source of silent bugs**.

### Rules (from trailing dimension):
1. Dimensions are compared **right to left**.
2. Dimensions must be **equal**, or one of them must be **1**.
3. A dimension of **1** is "stretched" to match the other."""),

        ("code", """# ✅ Valid broadcasting
a = torch.ones(3, 4)       # (3, 4)
b = torch.ones(4)          # (4,)  →  broadcast to (3, 4)
print(f"(3,4) + (4,): {(a + b).shape}")

c = torch.ones(3, 1)       # (3, 1) → broadcast to (3, 4)
print(f"(3,4) + (3,1): {(a + c).shape}")

# ⚠️ Corner Case: Silent shape mismatch!
x = torch.tensor([[1.], [2.], [3.]])   # (3, 1)
y = torch.tensor([10., 20., 30., 40.]) # (4,)
result = x + y  # (3, 1) + (4,) → broadcasts to (3, 4)!
print(f"\\n⚠️ (3,1) + (4,) = {result.shape}")
print(result)
# This might NOT be what you intended! You might have wanted (3,) + (3,)

# ❌ Invalid broadcasting
try:
    bad = torch.ones(3, 4) + torch.ones(2, 4)  # 3 ≠ 2, neither is 1
except RuntimeError as e:
    print(f"\\n❌ Error: {e}")"""),

        ("markdown", "## 3.4 — Reduction Operations"),

        ("code", """t = torch.tensor([[1., 2., 3.],
                     [4., 5., 6.]])

print(f"Sum all:      {t.sum()}")
print(f"Sum cols:     {t.sum(dim=0)}")     # Sum down rows → (3,)
print(f"Sum rows:     {t.sum(dim=1)}")     # Sum across cols → (2,)

print(f"Mean:         {t.mean()}")
print(f"Max:          {t.max()}")
print(f"Argmax (flat):{t.argmax()}")        # Index in flattened tensor
print(f"Argmax dim=1: {t.argmax(dim=1)}")   # Index of max per row

# ⚠️ Corner Case: keepdim
sum_no_keep = t.sum(dim=1)                # shape: (2,)
sum_keepdim = t.sum(dim=1, keepdim=True)  # shape: (2, 1)
print(f"\\nWithout keepdim: {sum_no_keep.shape}")
print(f"With keepdim:    {sum_keepdim.shape}")
# keepdim=True is essential when you want to broadcast the result
# back against the original tensor (e.g., normalization)

# Normalization example
normalized = t / t.sum(dim=1, keepdim=True)
print(f"\\nNormalized rows:\\n{normalized}")
print(f"Row sums check: {normalized.sum(dim=1)}")  # Should be [1, 1]"""),

        ("markdown", """## 3.5 — ⚠️ In-place Operations and Autograd"""),

        ("code", """# In-place operations end with _ (underscore)
x = torch.tensor([1., 2., 3.])
x.add_(10)       # In-place add
x.mul_(2)        # In-place multiply
print(f"After in-place: {x}")

# ⚠️ DANGER: In-place ops break autograd!
w = torch.tensor([1., 2., 3.], requires_grad=True)

# This would cause an error during backward():
# w.add_(1)  # ❌ RuntimeError: in-place operation on a leaf Variable

# ✅ Use out-of-place operations with autograd
w_new = w + 1   # Creates new tensor, gradient graph intact
print(f"Safe: {w_new}")"""),

        ("markdown", "## ✏️ Exercises"),

        ("code", """# Exercise 1: Given a (5, 3) matrix of exam scores (5 students, 3 exams),
# compute each student's average score using reduction ops.
scores = torch.randint(50, 100, (5, 3), dtype=torch.float32)
# YOUR CODE HERE


# Exercise 2: Manually normalize a tensor to have zero mean and unit std deviation
# (z = (x - mean) / std). Use keepdim properly.
data = torch.randn(4, 5)
# YOUR CODE HERE


# Exercise 3: What shapes will these broadcast to? Predict first, then verify:
# (5, 3, 1) + (1, 3, 4) = ?
# (2, 1, 6) + (1, 5, 1) = ?
# YOUR CODE HERE
"""),
    ])


# ═════════════════════════════════════════════════════════════
# CHAPTER 4 — AUTOGRAD
# ═════════════════════════════════════════════════════════════

def ch04():
    save("Ch04_Autograd_and_Gradients.ipynb", [
        ("markdown", """# 🔥 Chapter 4 — Autograd: Automatic Differentiation

---

### 🎯 What you'll learn
- How PyTorch tracks computations to compute gradients
- `requires_grad`, `.backward()`, `.grad`
- Gradient accumulation and zeroing
- `torch.no_grad()` and `detach()`
- ⚠️ Critical corner cases that cause silent bugs"""),

        ("markdown", """## 4.1 — The Computation Graph

PyTorch builds a **dynamic computation graph** as you perform operations. When you call `.backward()`, it traverses this graph in reverse to compute gradients via the **chain rule**."""),

        ("code", """import torch

# Create a tensor that tracks gradients
x = torch.tensor(3.0, requires_grad=True)

# Forward pass — build the computation graph
y = x ** 2 + 2 * x + 1   # y = x² + 2x + 1

# Backward pass — compute gradients
y.backward()

# dy/dx = 2x + 2 → at x=3: 2(3) + 2 = 8
print(f"x = {x}")
print(f"y = {y}")
print(f"dy/dx = {x.grad}")  # 8.0"""),

        ("markdown", """## 4.2 — ⚠️ Corner Case #1: Gradient Accumulation

Gradients are **accumulated** (added) each time `.backward()` is called. If you don't zero them, they keep growing!"""),

        ("code", """x = torch.tensor(2.0, requires_grad=True)

# First backward
y1 = x ** 2
y1.backward()
print(f"After 1st backward: grad = {x.grad}")  # 4.0 (2*2)

# ⚠️ Second backward WITHOUT zeroing
y2 = x ** 2
y2.backward()
print(f"After 2nd backward: grad = {x.grad}")  # 8.0! (accumulated: 4+4)

# ✅ Always zero gradients before backward
x.grad.zero_()
y3 = x ** 2
y3.backward()
print(f"After zeroing + backward: grad = {x.grad}")  # 4.0 ✅"""),

        ("markdown", """## 4.3 — Disabling Gradient Tracking

Use `torch.no_grad()` during inference (prediction) to:
1. Save memory (no computation graph stored)
2. Speed up computation"""),

        ("code", """x = torch.randn(1000, 1000, requires_grad=True)

# ❌ This tracks gradients (wastes memory during inference)
y = x * 2 + 3

# ✅ Method 1: torch.no_grad() context manager
with torch.no_grad():
    y = x * 2 + 3
    print(f"requires_grad inside no_grad: {y.requires_grad}")  # False

# ✅ Method 2: detach() — creates a tensor sharing data but without grad
y_detached = x.detach()
print(f"Detached requires_grad: {y_detached.requires_grad}")  # False

# ⚠️ Corner Case: detach() shares memory!
y_detached[0, 0] = 999
print(f"Original modified? {x[0, 0]}")  # YES — shared memory!
# Use .detach().clone() for a truly independent copy"""),

        ("markdown", """## 4.4 — ⚠️ Corner Case #2: Non-scalar Backward

`.backward()` only works directly on **scalar** tensors. For non-scalar outputs, you must provide a gradient argument."""),

        ("code", """x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x ** 2  # y is a vector, not a scalar!

# y.backward()  # ❌ RuntimeError: grad can be implicitly created only for scalar outputs

# ✅ Option 1: Reduce to scalar first
y.sum().backward()
print(f"Grad via sum: {x.grad}")  # [2, 4, 6]

# ✅ Option 2: Provide gradient tensor
x.grad.zero_()
y = x ** 2
y.backward(torch.ones_like(y))  # Equivalent to y.sum().backward()
print(f"Grad via ones: {x.grad}")  # [2, 4, 6]"""),

        ("markdown", """## 4.5 — Practical Gradient Descent Example"""),

        ("code", """# Minimize f(x) = (x - 5)² using gradient descent
x = torch.tensor(0.0, requires_grad=True)
learning_rate = 0.1

print(f"{'Step':>4} | {'x':>8} | {'f(x)':>8} | {'grad':>8}")
print("-" * 40)

for step in range(20):
    # Forward
    f = (x - 5) ** 2
    
    # Backward
    f.backward()
    
    # Update (must be inside no_grad!)
    with torch.no_grad():
        x -= learning_rate * x.grad
    
    # Zero gradients!
    x.grad.zero_()
    
    if step % 4 == 0:
        print(f"{step:4d} | {x.item():8.4f} | {f.item():8.4f} | {x.grad.item() if x.grad is not None else 0:8.4f}")

print(f"\\nFinal x = {x.item():.6f} (should approach 5.0)")"""),

        ("markdown", "## ✏️ Exercises"),

        ("code", """# Exercise 1: Compute the gradient of f(x, y) = x²y + y³ at x=2, y=3
# Expected: df/dx = 2xy = 12, df/dy = x² + 3y² = 31
# YOUR CODE HERE


# Exercise 2: Use gradient descent to find the minimum of f(x) = x⁴ - 4x² + 1
# Start from x = 3.0, learning rate = 0.01, 500 steps
# YOUR CODE HERE
"""),
    ])


# ═════════════════════════════════════════════════════════════
# CHAPTER 5 — DATASETS & DATALOADERS
# ═════════════════════════════════════════════════════════════

def ch05():
    save("Ch05_Datasets_and_DataLoaders.ipynb", [
        ("markdown", """# 🔥 Chapter 5 — Datasets, DataLoaders & Data Pipelines

---

### 🎯 What you'll learn
- `torch.utils.data.Dataset` and `DataLoader`
- Custom datasets for any data format
- Transforms and data augmentation
- Batching, shuffling, and worker processes
- ⚠️ Corner cases: worker memory, reproducibility, imbalanced data"""),

        ("markdown", "## 5.1 — The Dataset & DataLoader Pattern"),

        ("code", """import torch
from torch.utils.data import Dataset, DataLoader

# PyTorch's data loading has two core classes:
# 1. Dataset — defines HOW to access each sample
# 2. DataLoader — wraps Dataset for batching, shuffling, parallel loading

# Let's start with a built-in dataset
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples:     {len(test_dataset)}")
print(f"Sample shape:     {train_dataset[0][0].shape}")
print(f"Label:            {train_dataset[0][1]}")"""),

        ("markdown", "## 5.2 — Creating a Custom Dataset"),

        ("code", """import torch
from torch.utils.data import Dataset, DataLoader

class SyntheticDataset(Dataset):
    \"\"\"Custom dataset: y = 2x + 3 + noise\"\"\"
    
    def __init__(self, num_samples=1000):
        self.x = torch.randn(num_samples, 1)
        self.y = 2 * self.x + 3 + 0.1 * torch.randn(num_samples, 1)
    
    def __len__(self):
        # REQUIRED: return total number of samples
        return len(self.x)
    
    def __getitem__(self, idx):
        # REQUIRED: return a single sample (features, label)
        return self.x[idx], self.y[idx]

dataset = SyntheticDataset(1000)
print(f"Dataset size: {len(dataset)}")
print(f"Sample: x={dataset[0][0].item():.4f}, y={dataset[0][1].item():.4f}")"""),

        ("markdown", "## 5.3 — DataLoader: Batching, Shuffling, Workers"),

        ("code", """loader = DataLoader(
    dataset,
    batch_size=32,       # Samples per batch
    shuffle=True,        # Shuffle every epoch
    num_workers=0,       # Parallel data loading processes (0 = main process)
    drop_last=False,     # Drop last incomplete batch?
    pin_memory=True,     # Speed up CPU→GPU transfer (use when training on GPU)
)

# Iterate through batches
for batch_idx, (x_batch, y_batch) in enumerate(loader):
    if batch_idx == 0:
        print(f"Batch shape: x={x_batch.shape}, y={y_batch.shape}")
        print(f"First 5 x values: {x_batch[:5, 0].tolist()}")
    break

print(f"Total batches: {len(loader)}")"""),

        ("markdown", """## 5.4 — ⚠️ Corner Cases"""),

        ("code", """# ⚠️ Corner Case 1: num_workers > 0 on macOS/Windows
# On macOS, use 'fork' or keep num_workers=0 to avoid crashes.
# On Windows, DataLoader with num_workers > 0 requires:
#   if __name__ == '__main__':   (multiprocessing guard)

# ⚠️ Corner Case 2: Reproducibility
# DataLoader shuffling uses its own RNG. For reproducible results:
def seed_worker(worker_id):
    import numpy as np
    import random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

reproducible_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g,
)

# ⚠️ Corner Case 3: drop_last matters for BatchNorm!
# If last batch has only 1 sample, BatchNorm will crash (needs at least 2).
# Use drop_last=True during training when using BatchNorm.
print("Use drop_last=True with BatchNorm during training!")"""),

        ("code", """# ⚠️ Corner Case 4: Handling imbalanced datasets
from torch.utils.data import WeightedRandomSampler

# Suppose we have 90% class 0, 10% class 1
labels = torch.cat([torch.zeros(900), torch.ones(100)]).long()
class_counts = torch.bincount(labels).float()
class_weights = 1.0 / class_counts
sample_weights = class_weights[labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# ⚠️ Cannot use shuffle=True with a custom sampler!
# balanced_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
print(f"Class counts: {class_counts}")
print(f"Weights: {class_weights}")
print("WeightedRandomSampler balances mini-batches!")"""),

        ("markdown", "## ✏️ Exercises"),

        ("code", """# Exercise 1: Create a custom Dataset for the XOR problem:
# Inputs: (0,0), (0,1), (1,0), (1,1)  Labels: 0, 1, 1, 0
# Then wrap in a DataLoader with batch_size=2 and iterate.
# YOUR CODE HERE


# Exercise 2: Create a Dataset that reads lines from a text file.
# Each line has format "feature1,feature2,label".
# Write the CSV file first, then create the Dataset.
# YOUR CODE HERE
"""),
    ])


# ═════════════════════════════════════════════════════════════
# CHAPTER 6 — BUILDING NEURAL NETWORKS
# ═════════════════════════════════════════════════════════════

def ch06():
    save("Ch06_Building_Neural_Networks.ipynb", [
        ("markdown", """# 🔥 Chapter 6 — Building Neural Networks with `nn.Module`

---

### 🎯 What you'll learn
- `nn.Module` — the base class for all models
- Common layers: Linear, Conv2d, RNN, LSTM
- Activation functions and when to use each
- Model inspection, parameter counting
- ⚠️ Corner cases: weight initialization, forgetting `super().__init__()`"""),

        ("markdown", "## 6.1 — Your First Neural Network"),

        ("code", """import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()  # ⚠️ NEVER forget this!
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

model = SimpleNet(input_size=10, hidden_size=32, output_size=3)
print(model)

# Test with random input
x = torch.randn(5, 10)  # batch of 5, 10 features each
output = model(x)        # Calls forward() internally
print(f"\\nInput shape:  {x.shape}")
print(f"Output shape: {output.shape}")"""),

        ("markdown", """## 6.2 — ⚠️ Corner Case: NEVER Call `forward()` Directly!"""),

        ("code", """# ❌ DON'T: model.forward(x)
# ✅ DO:    model(x)

# Why? model(x) calls __call__ which:
# 1. Runs all registered hooks (for debugging, logging)
# 2. Handles gradient tracking
# 3. THEN calls forward()

# model.forward(x) skips hooks and can cause subtle bugs!
print("Always use model(x), never model.forward(x)!")"""),

        ("markdown", "## 6.3 — Using `nn.Sequential` for Simple Models"),

        ("code", """# When the model is a simple stack of layers, use Sequential
model_seq = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Dropout(0.2),       # Randomly zero 20% of inputs (regularization)
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 3),
)

print(model_seq)
output = model_seq(torch.randn(5, 10))
print(f"Output shape: {output.shape}")"""),

        ("markdown", "## 6.4 — Common Layers Reference"),

        ("code", """# ── Fully Connected (Linear) ──
linear = nn.Linear(in_features=128, out_features=64, bias=True)
print(f"Linear weight shape: {linear.weight.shape}")  # (64, 128)

# ── Convolutional (2D) ──
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
img = torch.randn(1, 3, 32, 32)  # (batch, channels, height, width)
print(f"Conv2d input:  {img.shape}")
print(f"Conv2d output: {conv(img).shape}")  # (1, 16, 32, 32) — same spatial size due to padding=1

# ── Batch Normalization ──
bn = nn.BatchNorm2d(num_features=16)  # Match channels of previous layer

# ── Pooling ──
pool = nn.MaxPool2d(kernel_size=2, stride=2)
print(f"After pool: {pool(conv(img)).shape}")  # (1, 16, 16, 16) — halved

# ── Recurrent (LSTM) ──
lstm = nn.LSTM(input_size=50, hidden_size=100, num_layers=2, batch_first=True)
seq = torch.randn(8, 20, 50)  # (batch, seq_len, features)
output, (hidden, cell) = lstm(seq)
print(f"LSTM output: {output.shape}")  # (8, 20, 100)"""),

        ("markdown", "## 6.5 — Activation Functions: When to Use What"),

        ("code", """import torch.nn.functional as F

x = torch.linspace(-5, 5, 100)

# ReLU — default choice for hidden layers
# ⚠️ Corner Case: "Dying ReLU" — neurons can permanently output 0
relu = torch.relu(x)

# LeakyReLU — fixes dying ReLU with small negative slope
leaky = F.leaky_relu(x, negative_slope=0.01)

# GELU — used in Transformers (GPT, BERT)
gelu = F.gelu(x)

# Sigmoid — output layer for binary classification (range 0-1)
# ⚠️ Corner Case: saturates at extremes → vanishing gradients
sigmoid = torch.sigmoid(x)

# Softmax — output layer for multi-class classification (sums to 1)
# ⚠️ Corner Case: use dim= parameter correctly!
logits = torch.randn(3, 5)
probs = F.softmax(logits, dim=1)   # Softmax over classes (dim=1), NOT batch (dim=0)!
print(f"Probs sum per sample: {probs.sum(dim=1)}")  # Should be [1, 1, 1]"""),

        ("markdown", "## 6.6 — Model Inspection & Parameter Counting"),

        ("code", """model = SimpleNet(784, 256, 10)

# List all parameters
for name, param in model.named_parameters():
    print(f"{name:20s} shape={str(param.shape):15s} requires_grad={param.requires_grad}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\\nTotal parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")"""),

        ("markdown", "## 6.7 — ⚠️ Weight Initialization Corner Cases"),

        ("code", """model = SimpleNet(784, 256, 10)

# ⚠️ Default init may not be optimal for deep networks
# Apply custom initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        # Kaiming (He) init — best for ReLU networks
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(init_weights)
print("Kaiming initialization applied!")

# Other common initializations:
# nn.init.xavier_uniform_(m.weight)   # Best for Sigmoid/Tanh
# nn.init.normal_(m.weight, mean=0, std=0.01)
# nn.init.constant_(m.bias, 0)"""),

        ("markdown", "## ✏️ Exercises"),

        ("code", """# Exercise 1: Build a CNN for 28x28 grayscale images (like MNIST)
# Architecture: Conv→ReLU→Pool→Conv→ReLU→Pool→Flatten→FC→FC
# Print the model and verify output shape with a dummy input.
# YOUR CODE HERE


# Exercise 2: Write a function that prints a table of all layer names,
# output shapes (given an input), and parameter counts.
# YOUR CODE HERE
"""),
    ])


# ═════════════════════════════════════════════════════════════
# CHAPTER 7 — TRAINING LOOP
# ═════════════════════════════════════════════════════════════

def ch07():
    save("Ch07_Training_Loop_Mastery.ipynb", [
        ("markdown", """# 🔥 Chapter 7 — The Training Loop: Loss, Optimizers & Training

---

### 🎯 What you'll learn
- Loss functions: when to use which
- Optimizers: SGD, Adam, AdamW
- The complete training loop pattern
- Validation loop and overfitting detection
- ⚠️ Corner cases: forgetting `model.train()`, gradient clipping, learning rate"""),

        ("markdown", "## 7.1 — Loss Functions"),

        ("code", """import torch
import torch.nn as nn

# ── Regression ──
mse = nn.MSELoss()       # Mean Squared Error
mae = nn.L1Loss()         # Mean Absolute Error

pred = torch.tensor([2.5, 3.0, 4.5])
true = torch.tensor([3.0, 3.0, 4.0])
print(f"MSE: {mse(pred, true):.4f}")
print(f"MAE: {mae(pred, true):.4f}")

# ── Binary Classification ──
bce = nn.BCEWithLogitsLoss()  # Includes sigmoid — more numerically stable!
logits = torch.tensor([0.5, -1.0, 2.0])
labels = torch.tensor([1.0, 0.0, 1.0])
print(f"\\nBCE: {bce(logits, labels):.4f}")
# ⚠️ Use BCEWithLogitsLoss, NOT BCELoss + sigmoid separately!

# ── Multi-class Classification ──
ce = nn.CrossEntropyLoss()  # Includes softmax internally!
logits = torch.randn(3, 5)          # 3 samples, 5 classes
labels = torch.tensor([0, 3, 2])    # Class indices (NOT one-hot!)
print(f"CrossEntropy: {ce(logits, labels):.4f}")

# ⚠️ Corner Case: CrossEntropyLoss expects:
#   - Input: raw logits (N, C), NOT softmax probabilities!
#   - Target: class INDICES (N,), NOT one-hot vectors!"""),

        ("markdown", "## 7.2 — Optimizers"),

        ("code", """model = nn.Linear(10, 3)

# SGD — simple, needs tuning
optim_sgd = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam — adaptive, great default
optim_adam = torch.optim.Adam(model.parameters(), lr=0.001)

# AdamW — Adam with decoupled weight decay (preferred for transformers)
optim_adamw = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# ⚠️ Corner Case: Adam vs AdamW weight decay
# Adam's weight_decay is coupled with the adaptive learning rate — this is wrong!
# AdamW decouples it — always prefer AdamW when using weight decay.
print("Prefer AdamW over Adam when using weight_decay!")"""),

        ("markdown", """## 7.3 — The Complete Training Loop

This is the **single most important pattern** in PyTorch."""),

        ("code", """import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Setup ──
torch.manual_seed(42)

# Synthetic data: y = 3x₁ + 2x₂ + 1
X = torch.randn(1000, 2)
y = 3 * X[:, 0] + 2 * X[:, 1] + 1 + 0.1 * torch.randn(1000)
y = y.unsqueeze(1)

# Split
X_train, X_val = X[:800], X[800:]
y_train, y_val = y[:800], y[800:]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

# Model
model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ── Training Loop ──
num_epochs = 20
for epoch in range(num_epochs):
    # ✅ MUST: Set model to training mode
    model.train()
    train_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        # 1. Forward pass
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        # 2. Backward pass
        optimizer.zero_grad()   # ⚠️ MUST zero gradients!
        loss.backward()
        
        # 3. Update weights
        optimizer.step()
        
        train_loss += loss.item() * X_batch.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # ── Validation Loop ──
    model.eval()   # ✅ MUST: Set model to eval mode
    val_loss = 0.0
    with torch.no_grad():   # ✅ MUST: disable gradients for validation
        for X_batch, y_batch in val_loader:
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= len(val_loader.dataset)
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

print("\\n✅ Training complete!")"""),

        ("markdown", """## 7.4 — ⚠️ Critical Corner Cases"""),

        ("code", """# ⚠️ Corner Case 1: model.train() vs model.eval()
# model.train() — enables Dropout and updates BatchNorm running stats
# model.eval()  — disables Dropout and freezes BatchNorm
# FORGETTING THIS IS THE #1 CAUSE OF "MY MODEL WORKS IN TRAINING BUT NOT INFERENCE"!

# ⚠️ Corner Case 2: optimizer.zero_grad() placement
# Option A (traditional): before loss.backward()
# Option B (modern): optimizer.zero_grad(set_to_none=True)  ← slightly faster
# set_to_none=True sets .grad to None instead of zeros (less memory)

# ⚠️ Corner Case 3: Gradient Clipping (prevents exploding gradients)
model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

x = torch.randn(1, 10)
loss = model(x).sum()
loss.backward()

# Clip gradient norm to max 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
print("Gradient clipping applied!")

# ⚠️ Corner Case 4: Moving model to device BEFORE creating optimizer
# model = model.to(device)         ← Do this FIRST
# optimizer = Adam(model.parameters())  ← THEN create optimizer
# If reversed, optimizer tracks CPU parameters but model is on GPU!"""),

        ("markdown", "## ✏️ Exercises"),

        ("code", """# Exercise 1: Modify the training loop above to:
# a) Add early stopping (stop if val_loss hasn't improved for 5 epochs)
# b) Save the best model weights
# YOUR CODE HERE


# Exercise 2: Train a model on synthetic classification data:
# 2 features, 3 classes. Use CrossEntropyLoss.
# YOUR CODE HERE
"""),
    ])


# ═════════════════════════════════════════════════════════════
# CHAPTER 8 — SAVING, LOADING & DEPLOYMENT
# ═════════════════════════════════════════════════════════════

def ch08():
    save("Ch08_Saving_Loading_and_Deployment.ipynb", [
        ("markdown", """# 🔥 Chapter 8 — Saving, Loading & Model Deployment

---

### 🎯 What you'll learn
- Saving and loading model weights (`state_dict`)
- Saving complete checkpoints (for resuming training)
- Exporting models (TorchScript, ONNX)
- ⚠️ Corner cases: loading on different devices, architecture mismatches"""),

        ("markdown", "## 8.1 — Saving & Loading `state_dict` (Recommended)"),

        ("code", """import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 3)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = MyModel()

# ✅ Save only the state_dict (weights + biases)
torch.save(model.state_dict(), 'model_weights.pth')
print("Model weights saved!")

# ✅ Load into a new model instance
model2 = MyModel()  # Must define architecture first!
model2.load_state_dict(torch.load('model_weights.pth'))
model2.eval()  # Set to eval mode for inference
print("Model weights loaded!")"""),

        ("markdown", """## 8.2 — ⚠️ Corner Case: Loading on a Different Device"""),

        ("code", """# ⚠️ If model was saved on GPU but loading on CPU:
# model.load_state_dict(torch.load('model.pth'))  # ❌ May crash!

# ✅ Always specify map_location
model3 = MyModel()
model3.load_state_dict(
    torch.load('model_weights.pth', map_location='cpu')  # Force load to CPU
)
print("Loaded with map_location='cpu'")

# If loading to GPU:
# torch.load('model.pth', map_location='cuda:0')

# For device-agnostic loading:
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.load('model.pth', map_location=device)"""),

        ("markdown", "## 8.3 — Full Checkpointing (Resume Training)"),

        ("code", """# Save everything needed to resume training
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

checkpoint = {
    'epoch': 10,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': 0.05,
    'val_loss': 0.08,
}
torch.save(checkpoint, 'checkpoint.pth')
print("Checkpoint saved!")

# Resume training
model_resumed = MyModel()
optimizer_resumed = torch.optim.Adam(model_resumed.parameters())

checkpoint = torch.load('checkpoint.pth', map_location='cpu')
model_resumed.load_state_dict(checkpoint['model_state_dict'])
optimizer_resumed.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
print(f"Resumed from epoch {start_epoch}")"""),

        ("markdown", """## 8.4 — ⚠️ Corner Case: `strict=False` for Partial Loading"""),

        ("code", """# When model architecture changed (e.g., added new layers)
class MyModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 3)
        self.fc3 = nn.Linear(3, 1)  # NEW layer!
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model_v2 = MyModelV2()

# ⚠️ strict=True (default) will FAIL because fc3 is missing from saved weights
# model_v2.load_state_dict(torch.load('model_weights.pth'))  # ❌ Error!

# ✅ strict=False loads matching keys, ignores missing/unexpected
incompatible = model_v2.load_state_dict(
    torch.load('model_weights.pth', map_location='cpu'),
    strict=False
)
print(f"Missing keys:    {incompatible.missing_keys}")
print(f"Unexpected keys: {incompatible.unexpected_keys}")"""),

        ("markdown", "## 8.5 — TorchScript: Export for Production"),

        ("code", """model = MyModel()
model.eval()

# Method 1: Tracing (for models without control flow)
example_input = torch.randn(1, 10)
traced = torch.jit.trace(model, example_input)
traced.save('model_traced.pt')

# Load without needing the class definition!
loaded = torch.jit.load('model_traced.pt')
output = loaded(example_input)
print(f"TorchScript output: {output.shape}")

# ⚠️ Corner Case: tracing doesn't capture if/else branches!
# Use torch.jit.script() for models with dynamic control flow."""),

        ("markdown", "## ✏️ Exercises"),

        ("code", """# Exercise 1: Write a function save_best_model(model, val_loss, best_loss, path)
# that only saves when val_loss improves.
# YOUR CODE HERE


# Exercise 2: Save a checkpoint, modify the model by removing a layer,
# load with strict=False, and verify which keys were missing.
# YOUR CODE HERE
"""),
    ])


# ═════════════════════════════════════════════════════════════
# CHAPTER 9 — ADVANCED TECHNIQUES & CORNER CASES
# ═════════════════════════════════════════════════════════════

def ch09():
    save("Ch09_Advanced_Techniques_and_Corner_Cases.ipynb", [
        ("markdown", """# 🔥 Chapter 9 — Advanced Techniques & Common Pitfalls

---

### 🎯 What you'll learn
- Learning rate schedulers
- Mixed precision training (AMP)
- Gradient accumulation for large effective batch sizes
- Freezing layers for transfer learning
- The complete pitfall checklist"""),

        ("markdown", "## 9.1 — Learning Rate Schedulers"),

        ("code", """import torch
import torch.nn as nn

model = nn.Linear(10, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Step LR: decay by factor every N epochs
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Cosine Annealing: smooth decay
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# ReduceOnPlateau: reduce when metric stops improving
scheduler3 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# ⚠️ Corner Case: scheduler.step() placement
# For most schedulers: call AFTER optimizer.step()
# For ReduceLROnPlateau: pass the metric!
# scheduler3.step(val_loss)  ← needs the monitored metric

# Visualize LR schedule
lrs = []
optimizer_viz = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler_viz = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_viz, T_max=50)
for epoch in range(50):
    lrs.append(optimizer_viz.param_groups[0]['lr'])
    optimizer_viz.step()
    scheduler_viz.step()
print(f"LR at epoch 0: {lrs[0]:.6f}")
print(f"LR at epoch 25: {lrs[25]:.6f}")
print(f"LR at epoch 49: {lrs[49]:.6f}")"""),

        ("markdown", "## 9.2 — Mixed Precision Training (AMP)\nUse `float16` where possible for ~2x speedup on compatible GPUs."),

        ("code", """from torch.cuda.amp import autocast, GradScaler

# ⚠️ AMP is most beneficial on NVIDIA GPUs with Tensor Cores (V100, A100, etc.)
# On CPU or MPS, it provides little to no benefit.

model = nn.Sequential(nn.Linear(100, 256), nn.ReLU(), nn.Linear(256, 10))
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()  # Scales loss to prevent float16 underflow

x = torch.randn(32, 100)
y = torch.randint(0, 10, (32,))

# ── AMP Training Step ──
# with autocast():  # Use autocast for forward pass
#     output = model(x)
#     loss = criterion(output, y)
# 
# scaler.scale(loss).backward()    # Scale loss before backward
# scaler.step(optimizer)           # Unscale and step
# scaler.update()                  # Update scaler for next iteration
# optimizer.zero_grad()

# ⚠️ Corner Case: Not all operations support float16
# Convolutions, matmuls → float16 (fast)
# Reductions (sum, softmax, loss) → float32 (accurate)
# autocast handles this automatically!
print("AMP pattern ready — use on GPU for speedup!")"""),

        ("markdown", "## 9.3 — Gradient Accumulation\nSimulate larger batch sizes when GPU memory is limited."),

        ("code", """model = nn.Linear(10, 1)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()

accumulation_steps = 4  # Effective batch = batch_size × accumulation_steps

for i in range(20):
    x = torch.randn(8, 10)  # Small batch (8)
    y = torch.randn(8, 1)
    
    loss = criterion(model(x), y)
    loss = loss / accumulation_steps  # ⚠️ Scale loss by accumulation steps!
    loss.backward()                    # Gradients accumulate
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()       # Update weights every N mini-batches
        optimizer.zero_grad()  # Reset gradients
        print(f"Step {i+1}: weight updated (effective batch = {8 * accumulation_steps})")

print("\\nGradient accumulation simulates batch_size=32 using batch_size=8!")"""),

        ("markdown", "## 9.4 — Freezing Layers (Transfer Learning)"),

        ("code", """model = nn.Sequential(
    nn.Linear(100, 64),   # Feature extractor
    nn.ReLU(),
    nn.Linear(64, 32),    # Feature extractor
    nn.ReLU(),
    nn.Linear(32, 10),    # Classifier head
)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze only the last layer (classifier head)
for param in model[4].parameters():
    param.requires_grad = True

# ✅ Only pass trainable params to optimizer!
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)

# Verify
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable} / {total} parameters")"""),

        ("markdown", """## 9.5 — 🔥 The Complete Pitfall Checklist

| # | Pitfall | Fix |
|---|---------|-----|
| 1 | Forgot `model.train()` / `model.eval()` | Always toggle before train/eval loops |
| 2 | Forgot `optimizer.zero_grad()` | Gradients accumulate silently → wrong updates |
| 3 | Forgot `torch.no_grad()` in validation | Wastes GPU memory, may OOM |
| 4 | Used `model.forward(x)` instead of `model(x)` | Skips hooks → silent bugs |
| 5 | Wrong `dim=` in softmax/loss | Softmax over wrong axis → garbage probabilities |
| 6 | CrossEntropyLoss with one-hot targets | Expects class indices, not one-hot |
| 7 | Applied softmax before CrossEntropyLoss | CE includes softmax → double softmax! |
| 8 | Created optimizer before `model.to(device)` | Optimizer tracks wrong device tensors |
| 9 | In-place ops on tensors with `requires_grad` | Breaks autograd computation graph |
| 10 | Not using `drop_last=True` with BatchNorm | Single-sample batches crash BatchNorm |
| 11 | float16 overflow | Values > 65504 become `inf` |
| 12 | Loading model without `map_location` | Crash when GPU→CPU or vice versa |
| 13 | Views share memory with original tensor | Modifying view modifies original |
| 14 | Non-contiguous tensor with `.view()` | Use `.reshape()` or `.contiguous().view()` |
| 15 | Not scaling loss in gradient accumulation | Gradients are N× too large |"""),

        ("markdown", "## ✏️ Exercises"),

        ("code", """# Exercise 1: Implement a training loop with:
# - Cosine annealing LR scheduler
# - Gradient accumulation (4 steps)
# - Gradient clipping (max_norm=1.0)
# YOUR CODE HERE


# Exercise 2: Take a pre-trained model, freeze all layers,
# replace the last layer, and fine-tune only the new layer.
# YOUR CODE HERE
"""),
    ])


# ═════════════════════════════════════════════════════════════
# CHAPTER 10 — COMPLETE AI PROJECT
# ═════════════════════════════════════════════════════════════

def ch10():
    save("Ch10_AI_Project_Image_Classifier.ipynb", [
        ("markdown", """# 🔥 Chapter 10 — Complete AI Project: MNIST Digit Classifier

---

### 🎯 End-to-End Project
In this final chapter, we bring everything together to build, train, evaluate, and deploy a **Convolutional Neural Network (CNN)** that classifies handwritten digits from the **MNIST dataset** with **>98% accuracy**.

### What this project covers:
1. Data loading with transforms and augmentation
2. CNN architecture design
3. Training loop with validation, early stopping, and LR scheduling
4. Model evaluation: accuracy, confusion matrix, per-class metrics
5. Inference on new data
6. Saving and loading the trained model

> ⚠️ This chapter applies **every concept and corner case** from Chapters 1–9."""),

        ("markdown", "## Step 1: Setup & Configuration"),

        ("code", """import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import json
from collections import defaultdict

# ── Reproducibility ──
torch.manual_seed(42)

# ── Device ──
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
print(f"🖥️  Device: {device}")

# ── Hyperparameters ──
CONFIG = {
    "batch_size": 64,
    "learning_rate": 0.001,
    "num_epochs": 15,
    "patience": 5,         # Early stopping patience
}
print(f"📋 Config: {json.dumps(CONFIG, indent=2)}")"""),

        ("markdown", "## Step 2: Data Pipeline"),

        ("code", """# ── Transforms ──
train_transform = transforms.Compose([
    transforms.RandomRotation(10),           # Data augmentation
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Slight shifts
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean & std
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    # ⚠️ No augmentation on test data!
])

# ── Datasets ──
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=test_transform)

# ── Split train → train + validation ──
train_size = int(0.85 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# ── DataLoaders ──
train_loader = DataLoader(train_subset, batch_size=CONFIG["batch_size"],
                          shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_subset, batch_size=256,
                        shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=256,
                         shuffle=False, num_workers=0)

print(f"📊 Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_dataset)}")
print(f"📦 Batch shape: {next(iter(train_loader))[0].shape}")"""),

        ("markdown", "## Step 3: CNN Architecture"),

        ("code", """class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # (1,28,28) → (32,28,28)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # (32,28,28) → (32,28,28)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # → (32,14,14)
            nn.Dropout2d(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → (64,14,14)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # → (64,14,14)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # → (64,7,7)
            nn.Dropout2d(0.25),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),                # → (64*7*7) = (3136)
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 10),          # 10 digit classes
            # ⚠️ No softmax here — CrossEntropyLoss includes it!
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = MNISTClassifier().to(device)

# Parameter count
total_params = sum(p.numel() for p in model.parameters())
print(f"🧠 Model Architecture:\\n{model}")
print(f"\\n📐 Total Parameters: {total_params:,}")

# Verify with dummy input
dummy = torch.randn(1, 1, 28, 28).to(device)
print(f"✅ Output shape: {model(dummy).shape}")  # Should be (1, 10)"""),

        ("markdown", "## Step 4: Training with Early Stopping"),

        ("code", """# ── Setup ──
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ── Training State ──
history = defaultdict(list)
best_val_loss = float('inf')
patience_counter = 0

print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val Acc':>8} | {'LR':>10} | {'Time':>6}")
print("-" * 65)

for epoch in range(1, CONFIG["num_epochs"] + 1):
    start_time = time.time()
    
    # ── Training ──
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item() * X_batch.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # ── Validation ──
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            val_loss += criterion(output, y_batch).item() * X_batch.size(0)
            _, predicted = output.max(dim=1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    
    val_loss /= len(val_loader.dataset)
    val_acc = correct / total
    lr = optimizer.param_groups[0]['lr']
    elapsed = time.time() - start_time
    
    # ── LR Scheduling ──
    scheduler.step(val_loss)
    
    # ── History ──
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"{epoch:5d} | {train_loss:10.6f} | {val_loss:10.6f} | {val_acc:7.2%} | {lr:10.6f} | {elapsed:5.1f}s")
    
    # ── Early Stopping ──
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_mnist_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= CONFIG["patience"]:
            print(f"\\n⏹️  Early stopping at epoch {epoch} (no improvement for {CONFIG['patience']} epochs)")
            break

print(f"\\n✅ Best validation loss: {best_val_loss:.6f}")"""),

        ("markdown", "## Step 5: Evaluation on Test Set"),

        ("code", """# Load best model
model.load_state_dict(torch.load('best_mnist_model.pth', map_location=device))
model.eval()

# ── Test Metrics ──
all_preds = []
all_labels = []
test_loss = 0.0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        test_loss += criterion(output, y_batch).item() * X_batch.size(0)
        _, preds = output.max(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y_batch.cpu().tolist())

test_loss /= len(test_loader.dataset)
test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

print(f"📊 Test Results:")
print(f"   Loss:     {test_loss:.6f}")
print(f"   Accuracy: {test_acc:.2%}")"""),

        ("code", """# ── Confusion Matrix (Manual — no sklearn needed) ──
num_classes = 10
confusion = [[0] * num_classes for _ in range(num_classes)]
for true, pred in zip(all_labels, all_preds):
    confusion[true][pred] += 1

# Print confusion matrix
print("\\n📊 Confusion Matrix:")
print(f"{'':>6}", end="")
for i in range(10):
    print(f"{i:>6}", end="")
print()
print("-" * 66)

for i in range(10):
    print(f"{i:>4} |", end="")
    for j in range(10):
        val = confusion[i][j]
        print(f"{val:>6}", end="")
    print()

# ── Per-class Accuracy ──
print("\\n📊 Per-Class Accuracy:")
for digit in range(10):
    correct = confusion[digit][digit]
    total = sum(confusion[digit])
    acc = correct / total if total > 0 else 0
    bar = "█" * int(acc * 20)
    print(f"  Digit {digit}: {acc:6.2%} {bar}")"""),

        ("markdown", "## Step 6: Single Image Inference"),

        ("code", """# ── Predict a single image ──
def predict_digit(model, image_tensor, device):
    \"\"\"Run inference on a single image and return prediction + confidence.\"\"\"
    model.eval()
    with torch.no_grad():
        # Ensure correct shape: (1, 1, 28, 28)
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        elif image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        confidence, predicted = probs.max(dim=1)
        
        return predicted.item(), confidence.item(), probs.squeeze().cpu()

# Test on samples from test set
print("🔍 Predictions on test samples:\\n")
for i in [0, 100, 500, 1000, 5000]:
    img, true_label = test_dataset[i]
    pred_label, confidence, probs = predict_digit(model, img, device)
    status = "✅" if pred_label == true_label else "❌"
    print(f"  Sample {i:>5}: True={true_label}, Predicted={pred_label}, "
          f"Confidence={confidence:.2%} {status}")"""),

        ("markdown", "## Step 7: Export for Production"),

        ("code", """# ── TorchScript Export ──
model.eval()
model_cpu = model.to('cpu')

example_input = torch.randn(1, 1, 28, 28)
traced_model = torch.jit.trace(model_cpu, example_input)
traced_model.save('mnist_classifier_traced.pt')

# Verify traced model
loaded_model = torch.jit.load('mnist_classifier_traced.pt')
output = loaded_model(example_input)
print(f"✅ TorchScript model saved and verified!")
print(f"   Output shape: {output.shape}")
print(f"   Predicted digit: {output.argmax(dim=1).item()}")"""),

        ("markdown", """## 🎓 Congratulations!

You've completed the full PyTorch course — from tensors to a production-ready AI model!

### What you built in this project:
- ✅ Data pipeline with augmentation and train/val/test splits
- ✅ CNN with BatchNorm, Dropout, and proper initialization
- ✅ Training loop with early stopping, LR scheduling, and gradient clipping
- ✅ Evaluation with confusion matrix and per-class metrics
- ✅ Single-image inference function
- ✅ TorchScript export for deployment

### Next Steps:
1. Try different architectures (ResNet, VGG)
2. Apply to a different dataset (CIFAR-10, Fashion-MNIST)
3. Add TensorBoard logging
4. Deploy with Flask/FastAPI + TorchServe"""),

        ("code", """# 🧹 Cleanup generated files
import os
for f in ['model_weights.pth', 'checkpoint.pth', 'model_traced.pt',
          'best_mnist_model.pth', 'mnist_classifier_traced.pt']:
    if os.path.exists(f):
        # os.remove(f)  # Uncomment to clean up
        pass
print("Run cleanup by uncommenting os.remove() above!")
"""),
    ])


# ═════════════════════════════════════════════════════════════
# GENERATE ALL
# ═════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("=" * 60)
    print("  🔥 Generating PyTorch Course Notebooks")
    print("=" * 60)

    ch01()
    ch02()
    ch03()
    ch04()
    ch05()
    ch06()
    ch07()
    ch08()
    ch09()
    ch10()

    print("\n" + "=" * 60)
    print(f"  ✅ All 10 notebooks created in '{OUTDIR}/'")
    print("=" * 60)


if __name__ == "__main__":
    main()
