"""
Step 1: Generate Probabilities Only
Saves probabilities so you can quickly test different thresholds without re-running inference
"""

import numpy as np
import pandas as pd
from pathlib import Path
from autogluon.tabular import TabularPredictor
import time

# Configuration
MODEL_DIR = Path('/drive1/nammt/MALLORN/MALLORN-Challenge/models/autogluon')
DATA_DIR = Path('/drive1/nammt/MALLORN/MALLORN-Challenge/data/processed')
OUTPUT_DIR = Path('/drive1/nammt/MALLORN/MALLORN-Challenge/outputs_cracked')

print("="*80)
print("STEP 1: GENERATE PROBABILITIES (One-Time Inference)")
print("="*80)

# Load trained model
print("\nLoading trained model...")
start_time = time.time()
predictor = TabularPredictor.load(str(MODEL_DIR / 'ag_multiclass_physics'))
print(f"✓ Model loaded in {time.time() - start_time:.1f}s")

# ============================================================================
# VALIDATION SET PROBABILITIES
# ============================================================================
print("\n" + "="*80)
print("Generating VALIDATION probabilities...")
print("="*80)

# Load and recreate validation split
train_processed = pd.read_csv(DATA_DIR / 'train_physics_processed.csv')
train_processed['base_id'] = train_processed['object_id'].str.split('_aug_').str[0]

from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
groups = train_processed['base_id'].values
X_temp = train_processed.drop(columns=['object_id', 'target'])

train_idx, val_idx = next(gkf.split(X_temp, groups=groups))

val_data = train_processed.iloc[val_idx].copy()
y_val_binary = (val_data['target'] == 1).astype(int).values

# Add SpecType
train_log = pd.read_csv(Path('/drive1/nammt/MALLORN/MALLORN-Challenge/data/raw/train_log.csv'))
val_data = val_data.merge(
    train_log[['object_id', 'SpecType']],
    left_on='base_id',
    right_on='object_id',
    how='left',
    suffixes=('', '_log')
)

# Generate validation predictions
print("Generating predictions...")
start_time = time.time()
val_probs_all = predictor.predict_proba(val_data.drop(columns=['object_id', 'target', 'base_id', 'object_id_log']))
y_val_prob_tde = val_probs_all['TDE'].values
print(f"✓ Validation predictions: {time.time() - start_time:.1f}s")

# Save validation probabilities
val_probs_df = pd.DataFrame({
    'object_id': val_data['object_id'].values,
    'true_label': y_val_binary,
    'tde_probability': y_val_prob_tde
})
val_probs_path = OUTPUT_DIR / 'validation_probabilities.csv'
val_probs_df.to_csv(val_probs_path, index=False)
print(f"✓ Saved: {val_probs_path}")

print(f"\nValidation stats:")
print(f"  Total: {len(y_val_binary)}")
print(f"  True TDEs: {y_val_binary.sum()}")
print(f"  Probability range: [{y_val_prob_tde.min():.4f}, {y_val_prob_tde.max():.4f}]")
print(f"  Mean probability: {y_val_prob_tde.mean():.4f}")

# ============================================================================
# TEST SET PROBABILITIES
# ============================================================================
print("\n" + "="*80)
print("Generating TEST probabilities...")
print("="*80)

# Load test data
test_ag = pd.read_csv(DATA_DIR / 'test_physics_processed.csv')
test_ids = test_ag['object_id'].values

print(f"Test set: {len(test_ids)} objects")
print("Generating predictions...")
start_time = time.time()
test_probs_all = predictor.predict_proba(test_ag.drop(columns=['object_id']))
test_probs = test_probs_all['TDE'].values
print(f"✓ Test predictions: {time.time() - start_time:.1f}s")

# Save test probabilities
test_probs_df = pd.DataFrame({
    'object_id': test_ids,
    'tde_probability': test_probs
})
test_probs_path = OUTPUT_DIR / 'test_probabilities.csv'
test_probs_df.to_csv(test_probs_path, index=False)
print(f"✓ Saved: {test_probs_path}")

print(f"\nTest stats:")
print(f"  Total: {len(test_probs)}")
print(f"  Probability range: [{test_probs.min():.4f}, {test_probs.max():.4f}]")
print(f"  Mean probability: {test_probs.mean():.4f}")
print(f"  Median probability: {np.median(test_probs):.4f}")

# Probability distribution
print(f"\nTest probability distribution:")
for percentile in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    val = np.percentile(test_probs, percentile)
    print(f"  P{percentile:2d}: {val:.4f}")

print("\n" + "="*80)
print("✓ PROBABILITIES SAVED!")
print("="*80)
print(f"\nGenerated files:")
print(f"  1. {val_probs_path.name}")
print(f"     - Use this to find optimal threshold")
print(f"  2. {test_probs_path.name}")
print(f"     - Use this to generate submissions")
print("\nNext: Run the threshold testing script to find best threshold")
print("="*80)