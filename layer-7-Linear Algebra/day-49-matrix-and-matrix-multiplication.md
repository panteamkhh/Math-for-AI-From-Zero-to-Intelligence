# Layer 7 — Day 49: Matrix & Matrix Multiplication

**Keywords:** Matrix · Matrix Multiplication

---

### 🎯 Intuition
A matrix is a table of numbers that can represent a transformation, a dataset, or a collection of vectors. Matrix multiplication means efficiently applying several linear transformations at once, or combining several vectors together.

### 💡 ML/AI Application
Every computation in a neural network — from the forward pass to batch processing on a GPU — is written as matrix multiplication, because GPUs are extremely optimized for exactly this kind of computation.

### 📝 Mental Exercise
Why is processing a batch of 32 samples as a single matrix multiplication much faster than processing each sample one at a time?

### 🧪 Hands-on
```python
import numpy as np
import time

n_samples, n_features, n_hidden = 32, 100, 256
X_batch = np.random.randn(n_samples, n_features)
W = np.random.randn(n_features, n_hidden)

# ONE-AT-A-TIME: loop over each sample, multiply individually
start = time.time()
outputs_loop = np.zeros((n_samples, n_hidden))
for i in range(n_samples):
    outputs_loop[i] = X_batch[i] @ W
loop_time = time.time() - start

# BATCHED: one single matrix multiplication for the entire batch
start = time.time()
outputs_batch = X_batch @ W
batch_time = time.time() - start

print(f"One-at-a-time loop: {loop_time*1000:.4f} ms")
print(f"Batched matmul:     {batch_time*1000:.4f} ms")
print("Results match:", np.allclose(outputs_loop, outputs_batch))
# Batched matrix multiplication lets the hardware use highly optimized,
# parallelized routines (BLAS on CPU, cuBLAS on GPU) instead of many small
# separate operations with loop overhead -- this gap gets even bigger on a real GPU.
```

---
⬅ [Day 48](day-48-dot-product-and-cosine-similarity.md) · [Layer 7 Summary](layer-7-summary-review.md) · [Day 50 →](day-50-linear-transformation.md)
