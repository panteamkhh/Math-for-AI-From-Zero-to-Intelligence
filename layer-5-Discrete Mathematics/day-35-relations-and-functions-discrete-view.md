# Layer 5 — Day 35: Relations & Functions, the Discrete View

**Keywords:** Relations · Functions (Discrete View)

---

### 🎯 Intuition
A relation is a connection between members of two sets (like "user X purchased product Y"). A function is a special kind of relation where each input has exactly one output.

### 💡 ML/AI Application
Knowledge Graphs and relational database tables are built directly on this concept of a "relation," and they underpin graph-based recommender systems.

### 📝 Mental Exercise
Why is "every user has one email" a function, but "every user purchased several products" is not?

### 🧪 Hands-on
```python
# A function: each key (user) maps to exactly ONE value (email)
user_email = {
    "alice": "alice@mail.com",
    "bob": "bob@mail.com",
    "carol": "carol@mail.com",
}
print("Function (user -> email):", user_email)

# A relation but NOT a function: each key (user) can map to MULTIPLE values (products)
user_purchases = {
    "alice": ["shoes", "hat"],
    "bob": ["laptop"],
    "carol": ["shoes", "phone", "charger"],
}
print("Relation, not a function (user -> products):", user_purchases)

# Why the distinction matters: a dict naturally enforces "function" behavior
# (one key -> one value), while representing a general "relation" requires
# storing a LIST of values per key, or a separate table of (user, product) pairs --
# exactly how knowledge graphs and relational databases store many-to-many relations.
```

---
⬅ [Day 34](day-34-logical-reasoning-and-sets.md) · [Layer 5 Summary](layer-5-summary-review.md) · [Day 36 →](day-36-addition-and-multiplication-principles.md)
