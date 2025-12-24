"""
Step 2: Fast Threshold Testing - TDE Count Focused
Loads pre-computed probabilities and tests different TDE counts
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from sklearn.metrics import (f1_score, precision_score, recall_score, 
                            confusion_matrix)

# Configuration
OUTPUT_DIR = Path('/drive1/nammt/MALLORN/MALLORN-Challenge/outputs_cracked')

print("="*80)
print("STEP 2: TDE COUNT ANALYSIS (Magic Number Optimization)")
print("="*80)

# Load pre-computed probabilities
print("\nLoading pre-computed probabilities...")
val_probs = pd.read_csv(OUTPUT_DIR / 'validation_probabilities.csv')
test_probs = pd.read_csv(OUTPUT_DIR / 'test_probabilities.csv')

y_val = val_probs['true_label'].values
val_prob = val_probs['tde_probability'].values
test_prob = test_probs['tde_probability'].values

print(f"✓ Validation: {len(val_probs)} samples ({y_val.sum()} TDEs)")
print(f"✓ Test: {len(test_probs)} samples")

# ============================================================================
# TDE COUNT ANALYSIS (Magic Number Search)
# ============================================================================
print("\n" + "="*80)
print("SEARCHING FOR OPTIMAL TDE COUNT")
print("="*80)

val_size = len(y_val)
test_size = len(test_prob)
true_val_tdes = y_val.sum()

print(f"\nValidation context:")
print(f"  Total samples: {val_size}")
print(f"  True TDEs: {true_val_tdes}")
print(f"  TDE rate: {true_val_tdes/val_size*100:.2f}%")

print(f"\nTest context:")
print(f"  Total samples: {test_size}")
print(f"  Scale factor: {test_size/val_size:.2f}x larger than validation")

print("\n" + "="*80)
print("VALIDATION PERFORMANCE AT DIFFERENT TDE COUNTS")
print("="*80)

print("\nTest    Val      Val     Val                    Effective")
print("TDEs    TDEs     F1      Prec    Recall  TP  FP  Threshold")
print("-" * 70)

results = []

# Test range from 250 to 400 TDEs
for test_tde_count in range(250, 401, 5):
    # Scale to validation size
    val_tde_count = int(test_tde_count * val_size / test_size)
    
    if val_tde_count < 10 or val_tde_count > val_size:
        continue
    
    # Rank-based prediction on validation
    val_ranks = rankdata(val_prob, method='ordinal')
    val_cutoff = val_size - val_tde_count + 1
    val_preds = (val_ranks >= val_cutoff).astype(int)
    
    # Metrics
    f1 = f1_score(y_val, val_preds)
    precision = precision_score(y_val, val_preds, zero_division=0)
    recall = recall_score(y_val, val_preds)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_val, val_preds).ravel()
    
    # Effective threshold on test set
    test_threshold = np.sort(test_prob)[-test_tde_count]
    
    print(f"{test_tde_count:4d}    {val_tde_count:3d}    {f1:.4f}   {precision:.3f}   {recall:.3f}   {tp:3d} {fp:3d}   {test_threshold:.4f}")
    
    results.append({
        'test_tde_count': test_tde_count,
        'val_tde_count': val_tde_count,
        'val_f1': f1,
        'val_precision': precision,
        'val_recall': recall,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'test_threshold': test_threshold
    })

# ============================================================================
# ANALYSIS & RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS & RECOMMENDATIONS")
print("="*80)

results_df = pd.DataFrame(results)

# Find best F1
best_f1_idx = results_df['val_f1'].idxmax()
best_f1_row = results_df.iloc[best_f1_idx]

print(f"\n1. BEST VALIDATION F1:")
print(f"   Target TDE count: {best_f1_row['test_tde_count']:.0f}")
print(f"   Validation F1: {best_f1_row['val_f1']:.4f}")
print(f"   Precision: {best_f1_row['val_precision']:.3f}")
print(f"   Recall: {best_f1_row['val_recall']:.3f}")
print(f"   Effective threshold: {best_f1_row['test_threshold']:.4f}")

# Find top 5 by F1
print("\n2. TOP 5 TDE COUNTS BY F1:")
print("\n   Rank  TDEs   Val-F1   Precision  Recall  Threshold")
print("   " + "-" * 55)
top5 = results_df.nlargest(5, 'val_f1')
for rank, (_, row) in enumerate(top5.iterrows(), 1):
    print(f"   {rank:2d}.   {row['test_tde_count']:4.0f}   {row['val_f1']:.4f}    {row['val_precision']:.3f}     {row['val_recall']:.3f}   {row['test_threshold']:.4f}")

# Precision-Recall tradeoff zones
print("\n3. STRATEGY ZONES:")

high_precision = results_df[results_df['val_precision'] >= 0.85]
if len(high_precision) > 0:
    best_high_prec = high_precision.loc[high_precision['val_f1'].idxmax()]
    print(f"\n   HIGH PRECISION (≥85%):")
    print(f"   → {best_high_prec['test_tde_count']:.0f} TDEs")
    print(f"     F1: {best_high_prec['val_f1']:.4f}, Prec: {best_high_prec['val_precision']:.3f}, Recall: {best_high_prec['val_recall']:.3f}")

high_recall = results_df[results_df['val_recall'] >= 0.85]
if len(high_recall) > 0:
    best_high_recall = high_recall.loc[high_recall['val_f1'].idxmax()]
    print(f"\n   HIGH RECALL (≥85%):")
    print(f"   → {best_high_recall['test_tde_count']:.0f} TDEs")
    print(f"     F1: {best_high_recall['val_f1']:.4f}, Prec: {best_high_recall['val_precision']:.3f}, Recall: {best_high_recall['val_recall']:.3f}")

balanced = results_df[(results_df['val_precision'] >= 0.75) & (results_df['val_recall'] >= 0.75)]
if len(balanced) > 0:
    best_balanced = balanced.loc[balanced['val_f1'].idxmax()]
    print(f"\n   BALANCED (Prec≥75%, Recall≥75%):")
    print(f"   → {best_balanced['test_tde_count']:.0f} TDEs")
    print(f"     F1: {best_balanced['val_f1']:.4f}, Prec: {best_balanced['val_precision']:.3f}, Recall: {best_balanced['val_recall']:.3f}")

# Compare to your previous result
print("\n4. YOUR PREVIOUS RESULT:")
print(f"   Magic 315: Test F1 = 64.57%")

prev_result = results_df[results_df['test_tde_count'] == 315]
if len(prev_result) > 0:
    prev = prev_result.iloc[0]
    print(f"   Validation F1: {prev['val_f1']:.4f}")
    print(f"   Precision: {prev['val_precision']:.3f}, Recall: {prev['val_recall']:.3f}")

# Suggest improvements
print("\n5. RECOMMENDED NEXT SUBMISSIONS:")

# Find counts near best F1
candidates = results_df.nlargest(10, 'val_f1')['test_tde_count'].values

print(f"\n   Based on validation analysis, try these in order:")
for i, count in enumerate(candidates[:5], 1):
    row = results_df[results_df['test_tde_count'] == count].iloc[0]
    print(f"   {i}. Magic {count:.0f} → Val F1: {row['val_f1']:.4f} (Prec: {row['val_precision']:.3f}, Recall: {row['val_recall']:.3f})")

# Save detailed results
results_path = OUTPUT_DIR / 'tde_count_analysis.csv'
results_df.to_csv(results_path, index=False)
print(f"\n✓ Detailed results saved: {results_path}")

# ============================================================================
# VISUALIZATION DATA
# ============================================================================
print("\n" + "="*80)
print("F1 CURVE")
print("="*80)

print("\nF1 scores across TDE counts (visual guide):")
print("\nTDEs:  ", end="")
for count in range(250, 401, 25):
    print(f"{count:5d}", end="")
print("\nF1:    ", end="")
for count in range(250, 401, 25):
    row = results_df[results_df['test_tde_count'] == count]
    if len(row) > 0:
        f1 = row.iloc[0]['val_f1']
        print(f"{f1:5.3f}", end="")
    else:
        print("  -  ", end="")
print("\n")

# Show sweet spot range
best_range = results_df.nlargest(10, 'val_f1')
min_count = best_range['test_tde_count'].min()
max_count = best_range['test_tde_count'].max()
print(f"Sweet spot range: {min_count:.0f} - {max_count:.0f} TDEs")
print(f"Your 315 is {'INSIDE' if 315 >= min_count and 315 <= max_count else 'OUTSIDE'} this range")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nNext: Use step3_generate_submissions.py with your chosen TDE counts")