"""
MALLORN TDE Classification with AutoGluon
==========================================
Complete pipeline with physics-corrected features, grouped CV, and multiclass strategy.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import joblib
from sklearn.metrics import (f1_score, roc_auc_score, precision_recall_curve,
                            confusion_matrix, classification_report)
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
REAL_BASE_DIR = Path('/drive1/nammt/MALLORN/MALLORN-Challenge')
DATA_DIR = REAL_BASE_DIR / 'data/processed'
MODEL_DIR = REAL_BASE_DIR / 'models/autogluon'
OUTPUT_DIR = REAL_BASE_DIR / 'outputs_cracked'

MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("="*80)
print("AutoGluon TDE Classification - Multiclass Physics Pipeline")
print("="*80)
print(f"Data: {DATA_DIR}")
print(f"Models: {MODEL_DIR}")
print()

# ============================================================================
# CELL 1: LOAD PHYSICS-PROCESSED DATA + SPECTYPE
# ============================================================================
print("="*80)
print("CELL 1: Loading Physics-Processed Data with SpecType")
print("="*80)

train_path = DATA_DIR / 'train_physics_processed.csv'
test_path = DATA_DIR / 'test_physics_processed.csv'

print(f"Loading {train_path}")
train_raw = pd.read_csv(train_path)

print(f"Loading {test_path}")
test_raw = pd.read_csv(test_path)

# --- FIX START: Correct Merge for Augmented Rows ---
# 1. Create base_id first so we can link augmented rows to original labels
train_raw['base_id'] = train_raw['object_id'].apply(lambda x: str(x).split('_aug_')[0])

# Load SpecType from original metadata
train_log = pd.read_csv(REAL_BASE_DIR / 'data/raw/train_log.csv')


# 3. Merge on base_id (matches original star) vs object_id (in log)
train_raw = train_raw.merge(
    train_log[['object_id', 'SpecType']], 
    left_on='base_id',   # Use the stripped ID
    right_on='object_id', # Match against original ID in log
    how='left',
    suffixes=('', '_log')
)

# 4. Clean up columns (drop the extra object_id_log if created)
if 'object_id_log' in train_raw.columns:
    train_raw = train_raw.drop(columns=['object_id_log'])


missing_labels = train_raw['SpecType'].isnull().sum()
if missing_labels > 0:
    print(f"⚠️ WARNING: Found {missing_labels} rows with missing SpecType!")
    # Critical: Drop rows with no label, otherwise AutoGluon crashes
    train_raw = train_raw.dropna(subset=['SpecType'])
else:
    print("✓ All rows have SpecType labels.")


print(f"\nTraining data: {train_raw.shape}")
print(f"Test data: {test_raw.shape}")

# Check distributions
print(f"\nTarget distribution:")
print(train_raw['target'].value_counts())
print(f"TDE ratio: {train_raw['target'].mean()*100:.2f}%")

print(f"\nSpecType distribution:")
print(train_raw['SpecType'].value_counts())
print()

# Verify SpecType-target mapping
print("SpecType vs Target mapping:")
print(train_raw.groupby('SpecType')['target'].mean())
print()

# ============================================================================
# CELL 2: PREPARE FOR GROUPED CV
# ============================================================================
print("="*80)
print("CELL 2: Creating Group IDs for CV")
print("="*80)

# Create base_id to prevent augmented twins from splitting across folds
train_raw['base_id'] = train_raw['object_id'].str.split('_aug_').str[0]

print(f"✓ Created group IDs")
print(f"  Unique objects: {train_raw['base_id'].nunique()}")
print(f"  Total samples: {len(train_raw)}")
print(f"  Augmentation factor: {len(train_raw) / train_raw['base_id'].nunique():.2f}x")
print()

# ============================================================================
# CELL 3: PREPARE DATA FOR AUTOGLUON (NO PREPROCESSING)
# ============================================================================
print("="*80)
print("CELL 3: Preparing Data for AutoGluon")
print("="*80)

# Drop metadata columns only (keep Z, EBV, SpecType)
drop_cols = ['is_augmented']
drop_cols = [c for c in drop_cols if c in train_raw.columns]

train_ag = train_raw.drop(columns=drop_cols)
test_ag = test_raw.copy()

# Ensure column alignment
test_ag = test_ag[[c for c in train_ag.columns if c in test_ag.columns and c != 'SpecType']]

print(f"AutoGluon training data: {train_ag.shape}")
print(f"  Features: {len([c for c in train_ag.columns if c not in ['object_id', 'target', 'base_id', 'SpecType']])}")
print(f"  Has SpecType: ✓")
print(f"  Has base_id for grouped CV: ✓")
print(f"\nAutoGluon test data: {test_ag.shape}")
print()

# ============================================================================
# CELL 4: TRAIN AUTOGLUON WITH MULTICLASS + GROUPED CV (V1.5.0 FIX)
# ============================================================================
print("="*80)
print("CELL 4: Training AutoGluon with Multiclass Strategy")
print("="*80)

from autogluon.tabular import TabularPredictor
from sklearn.model_selection import GroupKFold

# Configuration - MULTICLASS
LABEL = 'SpecType'
EVAL_METRIC = 'f1_macro'
TIME_LIMIT = 14400
PRESET = 'best_quality'

print(f"Configuration:")
print(f"  Label: {LABEL} (MULTICLASS)")
print(f"  Metric: {EVAL_METRIC}")
print(f"  Preset: {PRESET}")
print(f"  Time limit: {TIME_LIMIT}s ({TIME_LIMIT/60:.0f} min)")
print(f"  Grouped CV: Manual GroupKFold Split")
print()

# Prepare training data - drop metadata but keep base_id temporarily for splitting
train_ag_with_id = train_ag.drop(columns=['target', 'object_id'])

# --- MANUAL GROUP SPLIT FOR AUTOGLUON 1.5.0 ---
# Split 80/20 based on base_id groups to prevent twin leakage
print("Creating grouped train/validation split...")
gkf = GroupKFold(n_splits=5)
groups = train_ag['base_id'].values
train_idx, val_idx = next(gkf.split(train_ag_with_id, groups=groups))

train_data_fold = train_ag_with_id.iloc[train_idx].drop(columns=['base_id'])
tuning_data_fold = train_ag_with_id.iloc[val_idx].drop(columns=['base_id'])

# Verify no leakage
train_base_ids = set(train_ag.iloc[train_idx]['base_id'])
val_base_ids = set(train_ag.iloc[val_idx]['base_id'])
has_leakage = bool(train_base_ids & val_base_ids)

print(f"\nManual Group Split:")
print(f"  Training: {len(train_data_fold)} rows")
print(f"  Validation: {len(tuning_data_fold)} rows")
print(f"  Leakage check (should be False): {has_leakage}")

if has_leakage:
    print("  ⚠️  WARNING: Data leakage detected!")
else:
    print("  ✓ No leakage - twins kept separate")

# Check class distribution in splits
print(f"\nClass distribution in splits:")
print(f"  Train TDEs: {(train_data_fold[LABEL] == 'TDE').sum()}")
print(f"  Val TDEs: {(tuning_data_fold[LABEL] == 'TDE').sum()}")
print()

print("Starting multiclass training...")
print("AutoGluon will:")
print("  1. Train on 80% of data (grouped by base_id)")
print("  2. Validate on 20% with no twin leakage")
print("  3. Learn to distinguish TDE from AGN, SNe, etc.")
print("  4. Optimize for macro F1 (prioritizes rare classes)")
print()

# Initialize predictor
predictor = TabularPredictor(
    label=LABEL,
    eval_metric=EVAL_METRIC,
    path=str(MODEL_DIR / 'ag_multiclass_physics'),
    problem_type='multiclass',
    verbosity=2
)

# Fit with explicit tuning data
predictor.fit(
    train_data=train_data_fold,
    tuning_data=tuning_data_fold,  # Explicit validation prevents leakage
    presets=PRESET,
    time_limit=TIME_LIMIT,
    
    # System resources
    num_cpus=8,
    num_gpus=1,
    
    # Ensemble settings
    auto_stack=True,
    num_bag_folds=5,
    num_bag_sets=3,
    num_stack_levels=3,
    use_bag_holdout=True,  # Use tuning_data for validation
    
    # Keep best models
    keep_only_best=True,
    save_space=True,
)

print("\n" + "="*80)
print("✓ Training Complete!")
print("="*80)
print()

# ============================================================================
# CELL 5: EVALUATE PERFORMANCE
# ============================================================================
print("="*80)
print("CELL 5: Model Performance Evaluation")
print("="*80)

# Leaderboard (evaluated on tuning_data_fold automatically)
leaderboard = predictor.leaderboard(tuning_data_fold, silent=True)
print("\nTop 10 Models by Macro F1 Score:")
print(leaderboard[['model', 'score_val', 'pred_time_val', 'fit_time']].head(10).to_string(index=False))

best_model_name = predictor.model_best
print(f"\n✓ Best model: {best_model_name}")

# Detailed evaluation on validation set
val_eval = predictor.evaluate(tuning_data_fold, silent=True)
print(f"\nValidation set metrics (leak-free grouped split):")
for metric, value in val_eval.items():
    print(f"  {metric}: {value:.4f}")

# Get predictions on validation set
y_val_multiclass = tuning_data_fold[LABEL].values
y_val_pred_multiclass = predictor.predict(tuning_data_fold)
y_val_prob_all = predictor.predict_proba(tuning_data_fold)

# Convert to binary for TDE evaluation
y_val_binary = (y_val_multiclass == 'TDE').astype(int)
y_val_pred_binary = (y_val_pred_multiclass == 'TDE').astype(int)
y_val_prob_tde = y_val_prob_all['TDE'].values

# Binary metrics
print("\nBinary TDE metrics on validation set:")
print(f"  Binary F1: {f1_score(y_val_binary, y_val_pred_binary):.4f}")
print(f"  Binary AUC: {roc_auc_score(y_val_binary, y_val_prob_tde):.4f}")

# Confusion matrix (multiclass)
print("\nMulticlass Confusion Matrix (validation set):")
cm = confusion_matrix(y_val_multiclass, y_val_pred_multiclass)
print(cm)

print("\nMulticlass Classification Report (validation set):")
print(classification_report(y_val_multiclass, y_val_pred_multiclass))

# Feature importance
importance = predictor.feature_importance(tuning_data_fold)
print("\nTop 20 Most Important Features:")
print(importance.head(20).to_string())

importance.to_csv(OUTPUT_DIR / 'feature_importance_multiclass.csv')
print(f"\n✓ Saved feature importance to {OUTPUT_DIR / 'feature_importance_multiclass.csv'}")
print()

# ============================================================================
# CELL 6: THRESHOLD TUNING ON VALIDATION DATA
# ============================================================================
print("="*80)
print("CELL 6: Binary Threshold Tuning for TDE Detection")
print("="*80)

# Find optimal threshold using validation TDE probabilities
precisions, recalls, thresholds = precision_recall_curve(y_val_binary, y_val_prob_tde)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

print(f"Optimal TDE threshold: {optimal_threshold:.3f}")
print(f"Expected binary F1 at this threshold: {f1_scores[best_idx]:.4f}")

# Show F1 at different thresholds
print("\nBinary F1 scores at different thresholds:")
for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    preds_at_thresh = (y_val_prob_tde >= thresh).astype(int)
    f1_at_thresh = f1_score(y_val_binary, preds_at_thresh)
    tde_rate = preds_at_thresh.mean() * 100
    recall = (preds_at_thresh & y_val_binary).sum() / (y_val_binary.sum() + 1e-10)
    precision = (preds_at_thresh & y_val_binary).sum() / (preds_at_thresh.sum() + 1e-10)
    print(f"  {thresh:.1f}: F1={f1_at_thresh:.4f}, Recall={recall:.3f}, Precision={precision:.3f}, TDE rate={tde_rate:.2f}%")

print()

# ============================================================================
# CELL 7: GENERATE TEST PREDICTIONS
# ============================================================================
print("="*80)
print("CELL 7: Generating Test Predictions")
print("="*80)

# Get MULTICLASS probabilities
print("Generating multiclass predictions...")
test_probs_all = predictor.predict_proba(test_ag.drop(columns=['object_id']))

# Extract ONLY the TDE probability column
test_probs = test_probs_all['TDE'].values

print(f"\nMulticlass prediction summary:")
print(f"  Classes predicted: {list(test_probs_all.columns)}")
print(f"  TDE probability extracted: ✓")

# Apply optimal threshold
test_preds = (test_probs >= optimal_threshold).astype(int)

print(f"\n✓ Predictions complete")
print(f"\nTest set predictions (threshold={optimal_threshold:.3f}):")
print(f"  Total objects: {len(test_preds)}")
print(f"  Predicted TDEs: {test_preds.sum()}")
print(f"  Predicted Non-TDEs: {(test_preds == 0).sum()}")
print(f"  TDE detection rate: {test_preds.mean() * 100:.2f}%")

print(f"\nTDE probability distribution:")
print(f"  Mean: {test_probs.mean():.4f}")
print(f"  Std: {test_probs.std():.4f}")
print(f"  Min: {test_probs.min():.4f}")
print(f"  Max: {test_probs.max():.4f}")
print(f"  Median: {np.median(test_probs):.4f}")
print(f"  Q25: {np.percentile(test_probs, 25):.4f}")
print(f"  Q75: {np.percentile(test_probs, 75):.4f}")

print(f"\nThreshold analysis:")
print(f"  Prob > 0.9: {(test_probs > 0.9).sum()}")
print(f"  Prob > 0.7: {(test_probs > 0.7).sum()}")
print(f"  Prob > 0.5: {(test_probs > 0.5).sum()}")
print(f"  Prob > 0.3: {(test_probs > 0.3).sum()}")
print()

# ============================================================================
# CELL 8: CREATE SUBMISSION
# ============================================================================
print("="*80)
print("CELL 8: Creating Submission")
print("="*80)

test_ids = test_ag['object_id'].values

# Create submission
submission = pd.DataFrame({
    'object_id': test_ids,
    'prediction': test_preds
})

submission_path = OUTPUT_DIR / 'submission_multiclass_physics.csv'
submission.to_csv(submission_path, index=False)

print(f"✓ Submission saved to {submission_path}")
print(f"\nSubmission preview:")
print(submission.head(10))
print(f"\nClass distribution:")
print(submission['prediction'].value_counts())

# Save probabilities
probs_df = pd.DataFrame({
    'object_id': test_ids,
    'tde_probability': test_probs,
    'prediction': test_preds
})
probs_path = OUTPUT_DIR / 'probabilities_multiclass_physics.csv'
probs_df.to_csv(probs_path, index=False)

print(f"\n✓ Probabilities saved to {probs_path}")

# High-confidence predictions
print("\nTop 20 highest-confidence TDE predictions:")
top_tdes = probs_df.nlargest(20, 'tde_probability')
for _, row in top_tdes.iterrows():
    print(f"  {row['object_id']}: {row['tde_probability']:.4f}")

print()

# ============================================================================
# CELL 9: SUMMARY
# ============================================================================
print("="*80)
print("PIPELINE COMPLETE - SUMMARY")
print("="*80)

print(f"\n📊 Final Results:")
print(f"  Best model: {best_model_name}")
print(f"  Optimal threshold: {optimal_threshold:.3f}")

print(f"\n📁 Generated Files:")
print(f"  ✓ Submission: {submission_path}")
print(f"  ✓ Probabilities: {probs_path}")
print(f"  ✓ Feature importance: {OUTPUT_DIR / 'feature_importance_multiclass.csv'}")
print(f"  ✓ Models: {MODEL_DIR / 'ag_multiclass_physics'}")

print(f"\n🎯 Prediction Summary:")
print(f"  Test objects: {len(test_ids)}")
print(f"  Predicted TDEs: {test_preds.sum()} ({test_preds.mean()*100:.2f}%)")
print(f"  High confidence (>0.9): {(test_probs > 0.9).sum()}")

print("\n" + "="*80)
print("🏆 MULTICLASS STRATEGY ENABLED - Ready for Top Performance!")
print("="*80)