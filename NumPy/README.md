# NumPy: From Basics to Data Analysis 🔢

A comprehensive, hands-on NumPy course covering everything from array foundations and memory layouts to vectorization, broadcasting tricks, linear algebra, and pure-NumPy data analysis.

---

## 📖 Course Overview

### 🟢 Foundations & Mechanics (Chapters 1–4)
- **Chapter 1:** [Introduction & Array Basics](Ch01_Introduction_and_Array_Basics.ipynb) — Why NumPy is 50-100x faster than pure Python lists, array creation routines, attributes (`shape`, `ndim`, `dtype`, `itemsize`, `nbytes`), and memory layouts.
- **Chapter 2:** [Indexing, Slicing & Fancy Indexing](Ch02_Indexing_Slicing_Fancy_Indexing.ipynb) — Multi-dimensional slicing, boolean masks/filters, conditional assignment (`np.where`), and views vs. copies.
- **Chapter 3:** [Array Manipulation & Reshaping](Ch03_Array_Manipulation_Reshaping.ipynb) — `reshape`, `ravel` vs `flatten`, transpose/axes swapping, stacking (`vstack`, `hstack`, `concatenate`, `stack`), splitting, and dimension expansions.
- **Chapter 4:** [Broadcasting Rules & Corner Cases](Ch04_Broadcasting_Rules_and_Corner_Cases.ipynb) — The 3 core broadcasting rules, silent shape-mismatch bugs, outer products, and pairwise distance matrix computation.

### 🟡 Mathematical Operations & Advanced Tools (Chapters 5–7)
- **Chapter 5:** [Mathematical Operations & Universal Functions (ufuncs)](Ch05_Math_Operations_and_Ufuncs.ipynb) — Element-wise arithmetic, reductions along axes (`axis=0` vs `axis=1`), `keepdims`, floating point comparison (`np.allclose`), and Einstein summation (`np.einsum`).
- **Chapter 6:** [Linear Algebra with NumPy](Ch06_Linear_Algebra.ipynb) — Matrix multiplication (`@`, `np.matmul`), determinants, inverses, eigenvalues/eigenvectors, solving linear equation systems, and least-squares regression.
- **Chapter 7:** [Random Numbers & Statistics](Ch07_Random_Numbers_and_Statistics.ipynb) — Modern Generator API (`np.random.default_rng`), sampling distributions, descriptive metrics, correlation matrices, and a Monte Carlo simulation estimating π.

### 🔴 Applied Data Analysis (Chapter 8)
- **Chapter 8:** [🎯 Pure NumPy Data Analysis Project](Ch08_Data_Analysis_Project.ipynb) — End-to-end data analysis on a 1,000-record employee dataset **without pandas**:
  - File I/O (`np.savetxt`, `np.save`, `np.load`)
  - Column-wise statistical summaries & correlation analysis
  - Group-by aggregations via boolean indexing
  - Outlier detection using z-scores
  - Continuous data binning (`np.digitize`)
  - Sorting and top/bottom performer ranking (`np.argsort`)

---

## 🚀 Getting Started

```bash
# Install NumPy (if not already installed)
pip install numpy

# Launch Jupyter Notebook
jupyter notebook
```
