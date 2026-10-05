"""
Solar Panel Anomaly Detection — Imbalance Handling
====================================================
Class distribution (20,000 samples):
  No-Anomaly    : 10000 (50.0%)   ← dominant
  Cell          :  1877  (9.4%)
  Vegetation    :  1639  (8.2%)
  Diode         :  1499  (7.5%)
  Cell-Multi    :  1288  (6.4%)
  Shadowing     :  1056  (5.3%)
  Cracking      :   940  (4.7%)
  Offline-Module:   827  (4.1%)
  Hot-Spot      :   249  (1.2%)
  Hot-Spot-Multi:   246  (1.2%)
  Soiling       :   204  (1.0%)
  Diode-Multi   :   175  (0.9%)   ← rarest

Two strategies are provided:
  1. Class Weights  — tell the loss function to penalize minority errors more
  2. Oversampling   — resample the DataFrame so every class has equal representation
"""

import pandas as pd
import numpy as np
from collections import Counter


# ── 0. Load ──────────────────────────────────────────────────────────────────

df = pd.read_csv("anomaly_classes.csv")          # filename | anomaly_class
class_counts = Counter(df["anomaly_class"])
classes = sorted(class_counts.keys())
n_classes = len(classes)
label2idx = {c: i for i, c in enumerate(classes)}
df["label"] = df["anomaly_class"].map(label2idx)


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1 — CLASS WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════
# Concept: keep the dataset as-is but scale each class's contribution to
# the loss inversely to its frequency.  Rare classes contribute more per
# sample, so the model can't "cheat" by always predicting No-Anomaly.
#
# Formula (sklearn convention):
#   weight_c = total_samples / (n_classes * count_c)
#
# Use case: pass weight_tensor to nn.CrossEntropyLoss(weight=...) in PyTorch
#           or class_weight="balanced" in sklearn estimators.

total = len(df)

class_weights_dict = {
    cls: total / (n_classes * count)
    for cls, count in class_counts.items()
}

# Convert to a tensor ordered by label index (for PyTorch)
try:
    import torch
    weight_tensor = torch.tensor(
        [class_weights_dict[c] for c in classes],
        dtype=torch.float32
    )
    print("PyTorch weight tensor:")
    for c, w in zip(classes, weight_tensor.tolist()):
        print(f"  {c:<20} idx={label2idx[c]}  weight={w:.4f}")
except ImportError:
    print("PyTorch not installed — weights as dict:")
    for c, w in class_weights_dict.items():
        print(f"  {c:<20} weight={w:.4f}")

# ── PyTorch usage example ────────────────────────────────────────────────────
PYTORCH_USAGE = """
import torch.nn as nn

criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(device))

# Training loop stays the same — the loss now auto-penalises rare class errors.
loss = criterion(logits, targets)
loss.backward()
"""

# ── Sklearn usage example ────────────────────────────────────────────────────
SKLEARN_USAGE = """
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(class_weight="balanced")
clf.fit(X_train, y_train)
"""


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2 — OVERSAMPLING  (via WeightedRandomSampler in PyTorch)
# ══════════════════════════════════════════════════════════════════════════════
# Concept: during training, draw mini-batches so that each class appears
# roughly equally often — minority classes are sampled with replacement.
# The underlying CSV/dataset is unchanged; only the sampling probabilities differ.
#
# Per-sample weight  = 1 / count_of_its_class
# Then torch.utils.data.WeightedRandomSampler draws indices accordingly.

sample_weights = np.array([
    1.0 / class_counts[cls]
    for cls in df["anomaly_class"]
])

try:
    import torch
    from torch.utils.data import WeightedRandomSampler

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(df),          # draw same total number of samples per epoch
        replacement=True              # minority classes will repeat; that's the point
    )

    SAMPLER_USAGE = """
    from torch.utils.data import DataLoader

    # Pass sampler instead of shuffle=True
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=sampler      # <-- replaces shuffle=True
    )
    """
    print("\\nWeightedRandomSampler created successfully.")
    print("Expected samples per class per epoch (approx):", total // n_classes)

except ImportError:
    print("PyTorch not installed — sampler skipped.")


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2b — OVERSAMPLING via pandas (framework-agnostic)
# ══════════════════════════════════════════════════════════════════════════════
# Useful when you want a balanced CSV on disk, or for sklearn pipelines.
# Upsample every minority class to match the majority class count.

majority_count = max(class_counts.values())   # 10000

balanced_parts = []
for cls in classes:
    subset = df[df["anomaly_class"] == cls]
    if len(subset) < majority_count:
        subset = subset.sample(majority_count, replace=True, random_state=42)
    balanced_parts.append(subset)

df_balanced = pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)

df_balanced.to_csv("anomaly_classes_balanced.csv", index=False)

print(f"\nBalanced CSV written: {len(df_balanced)} rows ({n_classes} classes × {majority_count} each)")
print("New distribution:")
print(df_balanced["anomaly_class"].value_counts().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("""
┌─────────────────────────────────────────────────────────────────────┐
│  WHICH STRATEGY TO USE?                                             │
├──────────────────┬──────────────────────────────────────────────────┤
│ Class Weights    │ Fastest, no data duplication. Best starting      │
│                  │ point for CNNs. Pass weight= to CrossEntropyLoss.│
├──────────────────┼──────────────────────────────────────────────────┤
│ WeightedSampler  │ Batches stay balanced without storing extra data.│
│ (PyTorch)        │ Good when dataset fits in memory.                │
├──────────────────┼──────────────────────────────────────────────────┤
│ Pandas Oversample│ Produces a new balanced CSV. Framework-agnostic. │
│                  │ Watch for overfitting on duplicated rare samples. │
└──────────────────┴──────────────────────────────────────────────────┘

For this dataset (50% No-Anomaly, Diode-Multi at 0.9%) the recommended
approach is:  Class Weights  +  WeightedRandomSampler together.
- Weights fix the loss signal.
- Sampler ensures every mini-batch actually *sees* rare classes.
""")