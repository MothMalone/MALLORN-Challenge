"""
Quick Submission Generator - Computes Fresh Threshold
"""

from scipy.stats import rankdata
import numpy as np
import pandas as pd
from pathlib import Path
from autogluon.tabular import TabularPredictor
from sklearn.metrics import precision_recall_curve, f1_score
import sys

# Configuration
MODEL_DIR = Path('/drive1/nammt/MALLORN/MALLORN-Challenge/models/autogluon')
DATA_DIR = Path('/drive1/nammt/MALLORN/MALLORN-Challenge/data/processed')
OUTPUT_DIR = Path('/drive1/nammt/MALLORN/MALLORN-Challenge/outputs')

print("="*80)
print("QUICK SUBMISSION GENERATOR (with threshold calculation)")
print("="*80)

# Load trained model
print("\nLoading trained model...")
predictor = TabularPredictor.load(str(MODEL_DIR / 'ag_multiclass_physics'))
print("✓ Model loaded: WeightedEnsemble_L5")

# ============================================================================
# STEP 1: Compute optimal threshold on validation data
# ============================================================================
print("\n" + "="*80)
print("STEP 1: Computing Optimal Threshold from Validation Data")
print("="*80)

# We need to recreate the validation split to get the threshold
# Load training data
train_processed = pd.read_csv(DATA_DIR / 'train_physics_processed.csv')

# Recreate base_id for grouping
train_processed['base_id'] = train_processed['object_id'].str.split('_aug_').str[0]

# Recreate the same split (first fold of GroupKFold with n_splits=5)
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
groups = train_processed['base_id'].values
X_temp = train_processed.drop(columns=['object_id', 'target'])

train_idx, val_idx = next(gkf.split(X_temp, groups=groups))

# Get validation data
val_data = train_processed.iloc[val_idx].copy()
y_val_binary = (val_data['target'] == 1).astype(int).values

# Need to add SpecType for prediction
train_log = pd.read_csv(Path('/drive1/nammt/MALLORN/MALLORN-Challenge/data/raw/train_log.csv'))
val_data = val_data.merge(
    train_log[['object_id', 'SpecType']],
    left_on='base_id',
    right_on='object_id',
    how='left',
    suffixes=('', '_log')
)

# Get validation predictions
print("Generating validation predictions...")
val_probs_all = predictor.predict_proba(val_data.drop(columns=['object_id', 'target', 'base_id', 'object_id_log']))
y_val_prob_tde = val_probs_all['TDE'].values

# Find optimal threshold
precisions, recalls, thresholds = precision_recall_curve(y_val_binary, y_val_prob_tde)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

print(f"\n✓ Optimal threshold: {optimal_threshold:.4f}")
print(f"  Expected F1: {f1_scores[best_idx]:.4f}")

# Show threshold sensitivity
print("\nThreshold sensitivity analysis:")
for thresh in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
    preds = (y_val_prob_tde >= thresh).astype(int)
    f1 = f1_score(y_val_binary, preds)
    n_pred = preds.sum()
    print(f"  {thresh:.2f}: F1={f1:.4f}, predicts {n_pred} TDEs")

# ============================================================================
# STEP 2: Generate test predictions
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Generating Test Predictions")
print("="*80)

# Load test data
test_ag = pd.read_csv(DATA_DIR / 'test_physics_processed.csv')
test_ids = test_ag['object_id'].values

print("Generating predictions...")
test_probs_all = predictor.predict_proba(test_ag.drop(columns=['object_id']))
test_probs = test_probs_all['TDE'].values

print(f"\nTDE probability distribution:")
print(f"  Mean: {test_probs.mean():.4f}")
print(f"  Median: {np.median(test_probs):.4f}")
print(f"  Min: {test_probs.min():.4f}")
print(f"  Max: {test_probs.max():.4f}")

# ============================================================================
# STRATEGY 1: Validation-Optimized Threshold
# ============================================================================
print("\n" + "="*80)
print("STRATEGY 1: Validation F1-Optimized Threshold")
print("="*80)

test_preds_threshold = (test_probs >= optimal_threshold).astype(int)

print(f"\nThreshold: {optimal_threshold:.4f}")
print(f"  Predicted TDEs: {test_preds_threshold.sum()}")
print(f"  TDE rate: {test_preds_threshold.mean() * 100:.2f}%")

submission_threshold = pd.DataFrame({
    'object_id': test_ids,
    'prediction': test_preds_threshold
})

path_threshold = OUTPUT_DIR / f'submission_threshold_{optimal_threshold:.4f}.csv'
submission_threshold.to_csv(path_threshold, index=False)
print(f"\n✓ Saved: {path_threshold}")

# ============================================================================
# STRATEGY 2: Magic Number (315)
# ============================================================================
print("\n" + "="*80)
print("STRATEGY 2: Magic Number Ranking")
print("="*80)

MAGIC_NUMBER = 315

ranks = rankdata(test_probs, method='ordinal')
cutoff_rank = len(test_probs) - MAGIC_NUMBER + 1
test_preds_magic = (ranks >= cutoff_rank).astype(int)

magic_threshold = np.sort(test_probs)[-MAGIC_NUMBER]

print(f"\nMagic Number: {MAGIC_NUMBER}")
print(f"  Predicted TDEs: {test_preds_magic.sum()}")
print(f"  Effective threshold: {magic_threshold:.4f}")

submission_magic = pd.DataFrame({
    'object_id': test_ids,
    'prediction': test_preds_magic
})

path_magic = OUTPUT_DIR / 'submission_magic315.csv'
submission_magic.to_csv(path_magic, index=False)
print(f"\n✓ Saved: {path_magic}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*80)
print("COMPARISON & RECOMMENDATION")
print("="*80)

overlap = (test_preds_threshold & test_preds_magic).sum()

print(f"\nStrategy 1: {test_preds_threshold.sum()} TDEs")
print(f"Strategy 2: {test_preds_magic.sum()} TDEs")
print(f"Agreement: {overlap} TDEs ({overlap/MAGIC_NUMBER*100:.1f}%)")

if test_preds_threshold.sum() > 500:
    print(f"\n⚠️  Strategy 1 predicts {test_preds_threshold.sum()} TDEs (seems very high)")
    print("   → RECOMMEND: Submit Strategy 2 (Magic 315) FIRST")
elif test_preds_threshold.sum() < 250:
    print(f"\n✓ Strategy 1 predicts {test_preds_threshold.sum()} TDEs (conservative)")
    print("   → RECOMMEND: Submit Strategy 1 FIRST")
else:
    print(f"\n✓ Both strategies predict reasonable numbers")
    print("   → RECOMMEND: Submit Strategy 2 (Magic 315) FIRST")

# Save probabilities
probs_df = pd.DataFrame({
    'object_id': test_ids,
    'tde_probability': test_probs,
    'pred_threshold': test_preds_threshold,
    'pred_magic': test_preds_magic
})
probs_df.to_csv(OUTPUT_DIR / 'probabilities_comparison.csv', index=False)

print("\n" + "="*80)
print("READY FOR SUBMISSION!")
print("="*80)