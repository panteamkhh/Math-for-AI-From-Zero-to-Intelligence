# Layer 6 — Day 41: Conditional Probability & Bayes' Theorem

**Keywords:** Conditional Probability · Bayes' Theorem

---

### 🎯 Intuition
Conditional probability is the probability of an event "given that we already know something else." Bayes' theorem lets us flip that probability around and update our belief with new evidence.

### 💡 ML/AI Application
The Naive Bayes classifier is built directly from this theorem. More importantly, the very idea of "updating a belief with new evidence" underlies Bayesian learning, A/B testing, and how we interpret model results in the real world.

### 📝 Mental Exercise
If a medical test for a rare disease is 99% accurate, why might the true probability of having the disease given a positive result be much lower than 99%?

### 🧪 Hands-on
```python
# Classic Bayes' theorem example: the rare-disease trap
p_disease = 0.001            # prior: only 0.1% of people actually have the disease
p_positive_given_disease = 0.99      # test sensitivity
p_positive_given_healthy = 0.01      # false positive rate (test is 99% accurate -> 1% false positives)

p_healthy = 1 - p_disease

# P(positive) = P(positive|disease)*P(disease) + P(positive|healthy)*P(healthy)
p_positive = (p_positive_given_disease * p_disease) + (p_positive_given_healthy * p_healthy)

# Bayes' theorem: P(disease | positive) = P(positive|disease)*P(disease) / P(positive)
p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive

print(f"P(disease) prior:            {p_disease:.4f}")
print(f"P(positive test):             {p_positive:.4f}")
print(f"P(disease | positive test):   {p_disease_given_positive:.4f}")
# Even with a 99%-accurate test, because the disease is SO rare, most positive
# results still come from the huge pool of healthy people who got a false positive --
# so the true probability of disease given a positive test is much lower than 99%.
```

---
⬅ [Day 40](day-40-sample-space-event-probability.md) · [Layer 6 Summary](layer-6-summary-review.md) · [Day 42 →](day-42-random-variable-and-expected-value.md)
