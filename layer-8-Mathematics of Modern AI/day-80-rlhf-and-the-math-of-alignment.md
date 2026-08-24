# Layer 8 — Day 80: RLHF & the Math of Alignment

**Keywords:** Reward Model · Policy Gradient · RLHF

---

### 🎯 Intuition
RLHF (Reinforcement Learning from Human Feedback) trains a separate "reward model" to predict which of two model outputs a human would prefer, then nudges the main model's outputs to score higher under that reward — while a KL-divergence penalty keeps it from drifting too far from its original behavior.

### 💡 ML/AI Application
This is the math behind how modern assistants (like the model you're talking to right now) go from "predicts the next token" to "helpful and aligned with what people actually want." It directly reuses ideas from this whole layer: likelihood and MLE (Days 65–66) to train the reward model, gradients (Day 62) to update the policy, and KL Divergence (Day 61) to keep the fine-tuned model anchored to its base behavior.

### 📝 Mental Exercise
Why does RLHF fine-tuning typically include a KL-divergence penalty against the original ("reference") model, instead of letting the model optimize the reward model's score with no constraint at all?

### 🧪 Hands-on
```python
import numpy as np

np.random.seed(0)

def reward_model(response_quality_score):
    # A toy stand-in for a learned reward model's output
    return response_quality_score

def kl_penalty(new_probs, ref_probs):
    return np.sum(new_probs * np.log(new_probs / ref_probs))

# Two candidate "policies" (token probability distributions) for the same prompt
reference_policy = np.array([0.4, 0.35, 0.25])   # the original, pre-RLHF model
mildly_updated_policy = np.array([0.55, 0.30, 0.15])   # nudged toward higher reward
extremely_updated_policy = np.array([0.97, 0.02, 0.01])  # over-optimized toward reward

reward_mild = reward_model(0.7)
reward_extreme = reward_model(0.95)

kl_mild = kl_penalty(mildly_updated_policy, reference_policy)
kl_extreme = kl_penalty(extremely_updated_policy, reference_policy)

print(f"Mild update     -> reward: {reward_mild}, KL penalty: {kl_mild:.4f}")
print(f"Extreme update  -> reward: {reward_extreme}, KL penalty: {kl_extreme:.4f}")

total_objective_mild = reward_mild - 0.5 * kl_mild
total_objective_extreme = reward_extreme - 0.5 * kl_extreme
print(f"\nCombined objective (reward - KL), mild:    {total_objective_mild:.4f}")
print(f"Combined objective (reward - KL), extreme: {total_objective_extreme:.4f}")

# Without the KL term, the model could chase the reward model's score in ways
# that "game" it -- producing outputs that score high on the learned reward
# but are repetitive, degenerate, or have drifted away from coherent, broadly
# capable language. The KL penalty (Day 61) keeps the optimization anchored
# close to the reference model, trading a bit of raw reward for stability.
```

---

## 🎓 The 80-day path is complete

From sets and functions (Day 1) to RLHF and alignment (Day 80) — every layer built on the one before it. This is the full mathematical foundation: intuition first, formulas second, and a straight line from "what is a function" to "how does a modern AI assistant actually get trained to be helpful."

⬅ [Day 79](day-79-dropout-and-normalization.md) · [Layer 8 Summary](layer-8-summary-review.md)
