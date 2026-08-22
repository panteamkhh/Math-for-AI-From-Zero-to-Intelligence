# Layer 8 — Day 76: Similarity Search

**Keywords:** Similarity Search

---

### 🎯 Intuition
Similarity search means finding the closest vectors (embeddings) to a query vector within a large, high-dimensional space — an operational generalization of the distance and cosine-similarity concepts we saw earlier.

### 💡 ML/AI Application
This is the foundation of RAG (Retrieval-Augmented Generation) systems: when an LLM needs to find relevant information from a large database, it does so by searching for the nearest embeddings in a vector database.

### 📝 Mental Exercise
Why can't similarity search over millions of embeddings be done by brute-force comparing every single pair of vectors, and why does it need approximate algorithms (ANN)?

### 🧪 Hands-on
```python
import numpy as np
import time

np.random.seed(0)
n_vectors = 100_000
dim = 128

database = np.random.randn(n_vectors, dim)
query = np.random.randn(dim)

# BRUTE-FORCE: compare the query against EVERY vector in the database
start = time.time()
distances = np.linalg.norm(database - query, axis=1)
top_5_brute_force = np.argsort(distances)[:5]
brute_force_time = time.time() - start

print(f"Brute-force search over {n_vectors:,} vectors took {brute_force_time*1000:.2f} ms")
print(f"Top 5 nearest indices: {top_5_brute_force}")

# Now imagine this database with a BILLION vectors instead of 100,000 --
# brute-force cost scales LINEARLY with database size, so it would take
# roughly 10,000x longer. Approximate Nearest Neighbor (ANN) algorithms
# (like HNSW or IVF, used in vector databases) build an index structure
# that finds "close enough" neighbors in a tiny fraction of that time,
# trading a small amount of accuracy for a massive speedup.
print(f"\nAt this rate, a database 10,000x bigger would take roughly "
      f"{brute_force_time * 10000:.1f}s with brute force -- ANN indexes avoid this.")
```

---
⬅ [Day 75](day-75-positional-encoding.md) · [Layer 8 Summary](layer-8-summary-review.md) · [Day 77 →](day-77-scaling-laws.md)
