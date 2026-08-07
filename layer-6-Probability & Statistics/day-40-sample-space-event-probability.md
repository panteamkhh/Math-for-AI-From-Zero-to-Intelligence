# Layer 6 — Day 40: Sample Space, Event, Probability

**Keywords:** Sample Space · Event · Probability

---

### 🎯 Intuition
The sample space is every possible outcome that could happen. An event is a subset of those outcomes. Probability is a number between zero and one showing how "believable" that event is.

### 💡 ML/AI Application
The output of every modern classification model (softmax, sigmoid) is a probability distribution over the space of classes. Understanding probability precisely means understanding that a model isn't giving you a definite label — it's giving you a degree of confidence.

### 📝 Mental Exercise
If a model says P(spam)=0.7, what does that actually mean about the sample space of all similar emails?

### 🧪 Hands-on
```python
import numpy as np

def softmax(scores):
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)

# The "sample space" here is {spam, not_spam} -- every possible outcome
scores = np.array([1.2, 0.4])  # raw model scores for [spam, not_spam]
probs = softmax(scores)

print(f"P(spam) = {probs[0]:.2f}")
print(f"P(not_spam) = {probs[1]:.2f}")
print(f"Sum over the whole sample space = {probs.sum():.2f}  (must always be 1.0)")

# P(spam)=0.7 does NOT mean "this specific email is 70% spam".
# It means: among the space of all emails that produce features like this one,
# about 70% of them would truly be spam.
```

---
⬅ [Layer 6 Summary](layer-6-summary-review.md) · [Day 41 →](day-41-conditional-probability-and-bayes-theorem.md)
