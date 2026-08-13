# 🧮 Layer 7 — Summary Review

**Linear Algebra**

> A fast recap of all 9 days. Use this to refresh before an interview or right after finishing the full daily files. If a row doesn't ring a bell, go back to that day's file.

| Day | Topic | Core Idea (one line) | Where it shows up in ML/AI |
|---|---|---|---|
| [47](day-47-vector-norm-vector-distance.md) | Vector, Norm, Vector Distance | Norm = "length" of a vector, generalizing absolute value | L2 norm → ridge/KNN/embedding normalization; L1 norm → lasso/sparsity |
| [48](day-48-dot-product-and-cosine-similarity.md) | Dot Product & Cosine Similarity | Dot product = alignment × magnitude; cosine = alignment only | Attention in transformers, vector search, recommenders — all dot-product based |
| [49](day-49-matrix-and-matrix-multiplication.md) | Matrix & Matrix Multiplication | A table of numbers as transformation/dataset; matmul = many operations at once | The entire forward pass and GPU batch processing is matrix multiplication |
| [50](day-50-linear-transformation.md) | Linear Transformation | Rotates/stretches/compresses space, keeps lines straight, fixes the origin | Every Dense layer (pre-activation) is a linear transformation — the base of representation learning |
| [51](day-51-vector-space-and-orthogonality.md) | Vector Space & Orthogonality | All possible vectors under add/scale; orthogonal = fully unrelated | Orthogonal features are easy to learn separately; multicollinearity breaks regression |
| [52](day-52-basis-and-rank.md) | Basis & Rank | Smallest set of vectors spanning a space; rank = true independent dimensions | Low rank = redundant features — exactly what PCA/dimensionality reduction targets |
| [53](day-53-determinant.md) | Determinant | A scale factor for volume/area under a transformation; zero = collapsed dimension | Zero determinant → matrix not invertible → unstable/broken regression math |
| [54](day-54-eigenvalue-and-eigenvector.md) | Eigenvalue & Eigenvector | A vector whose direction survives a transformation unchanged; eigenvalue = its length change | The mathematical core of PCA — principal directions of variance in the data |
| [55](day-55-pca-and-svd.md) | PCA & SVD | Rotate to find the highest-spread directions and drop the rest; SVD generalizes to any matrix | Dimensionality reduction, visualization, recommender systems, image compression, classic NLP (LSA) |

---

## 🧠 90-second mental drill

Answer each in one sentence, without looking back:

1. **Day 47** — Why does L1 regularization tend to zero out weights while L2 only shrinks them?
2. **Day 48** — Why is cosine similarity preferred over the raw dot product when comparing documents of very different lengths?
3. **Day 49** — Why is batching 32 samples into one matrix multiplication so much faster than looping over them one at a time?
4. **Day 50** — Geometrically, what does it mean for a neural network layer to "transform the feature space"?
5. **Day 51** — Why can two nearly-identical features (like height in cm and height in m) destabilize linear regression?
6. **Day 52–53** — What does a low matrix rank tell you about a dataset, and why does a zero determinant break the closed-form regression formula?
7. **Day 54–55** — Why does PCA pick the direction with the largest eigenvalue first, and why is that choice mathematically the least lossy?

<details>
<summary>Show quick answers</summary>

1. L1's penalty applies constant "pressure" toward zero regardless of how small a weight already is, so optimization can push it all the way to exactly 0; L2's penalty weakens as a weight shrinks (it's quadratic), so it keeps shrinking weights smoothly without ever fully zeroing them out.
2. Cosine similarity measures only the angle (direction) between vectors, ignoring magnitude, so a long document and a short one that share the same topic still register as highly similar — the raw dot product would be thrown off by the longer document's larger numbers.
3. Batched matrix multiplication lets hardware (BLAS/cuBLAS, especially on a GPU) use highly parallelized, optimized routines instead of many small separate operations with per-iteration loop overhead.
4. It means the layer applies a linear transformation that moves, rotates, and rescales the "cloud" of data points into a new space with a different shape — ideally one where the patterns you care about become easier to separate.
5. Because the two features carry almost identical, highly correlated information, the regression math can't cleanly attribute effect to one over the other — small changes in the data can cause the individual coefficients to swing wildly even though the model's overall predictions barely change.
6. A low rank means many of the dataset's columns are redundant — the data has far fewer truly independent dimensions of information than columns; a zero determinant means the feature matrix isn't invertible, so the closed-form regression formula (which requires inverting a matrix built from the features) simply cannot be solved, or becomes numerically unstable.
7. The eigenvalue directly measures how much variance (information) lies along its eigenvector's direction, so picking the largest eigenvalue first captures the most possible spread in the data; any other choice of direction would necessarily retain less of the data's total variance, making it strictly more lossy.

</details>

---

## 🗺️ Layer 7 concept map (in one paragraph)

This layer is the **native language of deep learning** — everything from here on is written in vectors and matrices. Vectors, norms, and distance (Day 47) give you ways to measure size and closeness, which regularization and KNN depend on directly. The dot product and cosine similarity (Day 48) are arguably the single most-used operation in modern AI, powering attention, search, and recommendations. Matrices and matrix multiplication (Day 49) are how every neural network computation actually gets executed, especially on GPUs built for exactly this. Linear transformations (Day 50) reframe a network layer as literally reshaping space — the geometric heart of representation learning. Vector spaces and orthogonality (Day 51) explain why redundant, correlated features destabilize models, while basis and rank (Day 52) quantify exactly how much of that redundancy exists in a dataset. The determinant (Day 53) tells you when a transformation has collapsed information beyond recovery — precisely when matrix inversion (and closed-form regression) breaks. And eigenvalues/eigenvectors (Day 54) lead directly into PCA and SVD (Day 55): finding the directions in your data that hold the most information, and using that to compress, visualize, or factorize high-dimensional data with minimal loss.

---
**Full days:** [47](day-47-vector-norm-vector-distance.md) · [48](day-48-dot-product-and-cosine-similarity.md) · [49](day-49-matrix-and-matrix-multiplication.md) · [50](day-50-linear-transformation.md) · [51](day-51-vector-space-and-orthogonality.md) · [52](day-52-basis-and-rank.md) · [53](day-53-determinant.md) · [54](day-54-eigenvalue-and-eigenvector.md) · [55](day-55-pca-and-svd.md)

**Next layer:** Layer 8 — Mathematics of Modern AI →
