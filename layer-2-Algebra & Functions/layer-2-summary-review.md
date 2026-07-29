# 📈 Layer 2 — Summary Review

**Algebra & Functions**

> A fast recap of all 15 days. Use this to refresh before an interview or right after finishing the full daily files. If a row doesn't ring a bell, go back to that day's file.

| Day | Topic | Core Idea (one line) | Where it shows up in ML/AI |
|---|---|---|---|
| [6](day-06-function-as-a-machine.md) | Function as a Machine | One input → exactly one output, like a processing machine | A model is `f(features) = prediction`; a network is a chain of such functions (forward pass) |
| [7](day-07-domain-and-range.md) | Domain and Range | Domain = allowed inputs; Range = possible outputs | Activation function choice (sigmoid, tanh, ReLU) depends entirely on the range the target needs |
| [8](day-08-function-representation-and-graphs.md) | Function Representation & Graphs | Formula, table, or graph — the graph lets you "see" behavior | Reading loss curves, feature distributions, decision boundaries = "debugging with your eyes" |
| [9](day-09-increasing-decreasing-max-min-symmetry-periodicity.md) | Increasing/Decreasing, Max/Min, Symmetry, Periodicity | Direction of change + turning points | Gradient Descent moves downhill toward a minimum; risk of getting stuck at a local minimum |
| [10](day-10-linear-equations.md) | Linear Equations | `y = ax + b` — constant-ratio relationship | Linear regression is this equation with many features and learned coefficients |
| [11](day-11-quadratic-equations.md) | Quadratic Equations | Bowl-shaped curve, one minimum/maximum | MSE-style losses are quadratic → convex → gradient descent finds the minimum reliably |
| [12](day-12-inequalities-and-geometric-interpretation.md) | Inequalities & Geometric Interpretation | A region of solutions, not one exact point | Decision boundaries in classifiers (and SVMs) are exactly this: `w·x + b > 0` vs `< 0` |
| [13](day-13-linear-function.md) | Linear Function | Straight line, constant slope everywhere | Stacked linear layers without activation collapse into one linear layer — why non-linearity matters |
| [14](day-14-quadratic-function.md) | Quadratic Function | Symmetric curve, non-constant rate of change | L2 regularization squares the weights → penalizes large weights sharply |
| [15](day-15-absolute-value-function.md) | Absolute Value Function | "V" shape, distance from zero, sharp point at 0 | MAE loss, L1 regularization → sparsity (can push weights to exact zero) |
| [16](day-16-exponential-function.md) | Exponential Function | Always positive, growth accelerates without limit | Softmax and Sigmoid rely on `e^x` to guarantee positive outputs |
| [17](day-17-logarithmic-function.md) | Logarithmic Function | Inverse of exponential; compresses large numbers | Cross-Entropy Loss uses `-log(p)` — confidently wrong predictions get punished hard |
| [18](day-18-function-composition.md) | Function Composition | Output of one function feeds the next, like an assembly line | Deep networks are long compositions of layer functions; backprop uses the Chain Rule through them |
| [19](day-19-inverse-function.md) | Inverse Function | The "return trip" — undoes what the original function did | Un-normalizing predictions (`z*std + mean`); invertible functions power generative flow models |
| [20](day-20-modeling-with-functions.md) | Modeling with Functions | Pick the right function family to match the data's real behavior | Choosing model architecture = choosing a function family (linear vs. tree vs. deep net) |

---

## 🧠 90-second mental drill

Answer each in one sentence, without looking back:

1. **Day 6–7** — Why must the model's output layer match the range the target actually needs?
2. **Day 8–9** — What does an oscillating loss curve usually tell you about the learning rate, and what does Gradient Descent risk with multiple local minima?
3. **Day 10–12** — What does the slope coefficient `a` mean in a linear model, and how is a decision boundary really just an inequality?
4. **Day 13–15** — Why does stacking linear layers without activation add no power, and why does L1 push weights to exact zero while L2 doesn't?
5. **Day 16–17** — Why is `e^x` essential for turning scores into probabilities, and why does `-log(p)` punish confident wrong answers so hard?
6. **Day 18–20** — How does function composition relate to backpropagation's chain rule, and why does using a linear model on non-linear data cause underfitting?

<details>
<summary>Show quick answers</summary>

1. Because each activation function has a fixed range (sigmoid → (0,1), tanh → (-1,1), ReLU → [0,∞)), and using the wrong one makes the output meaningless for the task (e.g., ReLU can't represent a probability).
2. An oscillating curve usually means the learning rate is too large, causing overshoot; with several local minima, gradient descent — which only looks at the local neighborhood — can get trapped in one that isn't the global best.
3. `a` is how much the output changes per one-unit increase in the input (e.g., $ per year of experience); a decision boundary is the line `w·x + b = 0` where the "inequality" `w·x+b > 0` vs `< 0` decides the class on either side.
4. Composing linear functions always collapses back into one linear function, so depth alone adds nothing without non-linearity; L1's sharp corner at zero makes it easy to snap a weight to exactly 0, while L2's smooth quadratic curve only shrinks weights without fully zeroing them.
5. `e^x` is always positive for any real input, so exponentiating scores before normalizing guarantees valid, positive probabilities; `-log(p)` grows explosively as `p` approaches 0, so a confident-but-wrong prediction gets a much larger penalty than an unsure one.
6. Composition means `y = g(f(x))`; the chain rule lets backprop compute how a change deep in the network affects the final loss by multiplying local derivatives layer by layer; a linear model can't bend to fit a non-linear pattern, so it underfits and both training and test error stay high.

</details>

---

## 🗺️ Layer 2 concept map (in one paragraph)

This layer is about **functions as the basic unit of computation in ML.** A model is just a function (Day 6) with a specific domain and range (Day 7) that we can visualize (Day 8) and whose shape — increasing, decreasing, with minima — determines how optimization behaves (Day 9). Linear and quadratic equations (Days 10–12) give us the simplest, most analyzable relationships and show up everywhere from regression to decision boundaries. The specific function *shapes* — linear, quadratic, absolute value, exponential, logarithmic (Days 13–17) — are the actual building blocks inside every activation function and loss function you'll use. Composition (Day 18) is what a deep network *is*: layers chained together, differentiated end-to-end via the chain rule. Inversion (Day 19) is what lets you undo preprocessing to get predictions back into a human-readable scale. And modeling with functions (Day 20) ties it together: picking a model architecture is really picking which family of functions you believe best matches your data's true shape.

---
**Full days:** [6](day-06-function-as-a-machine.md) · [7](day-07-domain-and-range.md) · [8](day-08-function-representation-and-graphs.md) · [9](day-09-increasing-decreasing-max-min-symmetry-periodicity.md) · [10](day-10-linear-equations.md) · [11](day-11-quadratic-equations.md) · [12](day-12-inequalities-and-geometric-interpretation.md) · [13](day-13-linear-function.md) · [14](day-14-quadratic-function.md) · [15](day-15-absolute-value-function.md) · [16](day-16-exponential-function.md) · [17](day-17-logarithmic-function.md) · [18](day-18-function-composition.md) · [19](day-19-inverse-function.md) · [20](day-20-modeling-with-functions.md)

**Next layer:** Layer 3 — Geometry & Analytic Geometry →
