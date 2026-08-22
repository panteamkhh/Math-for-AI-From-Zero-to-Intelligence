# Layer 8 — Day 75: Positional Encoding

**Keywords:** Positional Encoding

---

### 🎯 Intuition
Because the Attention mechanism on its own has no sense of word order (to it, "the cat saw the dog" looks the same as "the dog saw the cat"), Positional Encoding adds information about each word's "position" into its embedding.

### 💡 ML/AI Application
Without positional encoding, Transformers couldn't understand word order in a sentence or token order in a sequence — this technique (often using sine/cosine functions) lets the model understand meaning and order at the same time.

### 📝 Mental Exercise
Why can Attention alone (without positional encoding) not tell the difference between "Ali hit Ahmad" and "Ahmad hit Ali"?

### 🧪 Hands-on
```python
import numpy as np

def get_positional_encoding(seq_len, d_model):
    positions = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / d_model)
    angles = positions * angle_rates
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe

# "Ali" and "Ahmad" appear as the SAME word embedding regardless of position --
# attention alone treats a "bag of words" the same no matter the order.
word_embedding_ali = np.array([1.0, 0.5, 0.2, 0.8])
word_embedding_ahmad = np.array([0.9, 0.4, 0.3, 0.7])

pe = get_positional_encoding(seq_len=3, d_model=4)
print("Positional encoding for positions 0, 1, 2:\n", np.round(pe, 3))

# "Ali" at position 0 vs "Ali" at position 2 -- now they get DIFFERENT
# combined vectors, purely because of where they sit in the sentence:
ali_at_pos0 = word_embedding_ali + pe[0]
ali_at_pos2 = word_embedding_ali + pe[2]
print("\n'Ali' embedding + position 0:", np.round(ali_at_pos0, 3))
print("'Ali' embedding + position 2:", np.round(ali_at_pos2, 3))
# Without this addition, swapping the ORDER of "Ali" and "Ahmad" in a sentence
# would leave the set of token embeddings fed to attention completely
# unchanged -- positional encoding is what breaks that order-blindness.
```

---
⬅ [Day 74](day-74-query-key-value.md) · [Layer 8 Summary](layer-8-summary-review.md) · [Day 76 →](day-76-similarity-search.md)
