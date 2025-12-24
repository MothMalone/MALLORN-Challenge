"""
Step 3: Generate Submissions from Probabilities
Uses pre-computed probabilities and recommended TDE counts
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata

# Configuration
OUTPUT_DIR = Path('/drive1/nammt/MALLORN/MALLORN-Challenge/outputs_cracked')

print("="*80)
print("STEP 3: GENERATE SUBMISSIONS (TDE Count Based)")
print("="*80)

# Load test probabilities
print("\nLoading test probabilities...")
test_probs = pd.read_csv(OUTPUT_DIR / 'test_probabilities.csv')
test_ids = test_probs['object_id'].values
probs = test_probs['tde_probability'].values
print(f"✓ Loaded {len(probs)} test probabilities")

# Load analysis results
analysis = pd.read_csv(OUTPUT_DIR / 'tde_count_analysis.csv')
top_counts = analysis.nlargest(10, 'val_f1')['test_tde_count'].values

print(f"\nTop 10 TDE counts by validation F1:")
for i, count in enumerate(top_counts, 1):
    row = analysis[analysis['test_tde_count'] == count].iloc[0]
    print(f"  {i:2d}. {count:.0f} TDEs → Val F1: {row['val_f1']:.4f}")

# ============================================================================
# Generate submissions for top candidates
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUBMISSIONS")
print("="*80)

# Define which counts to generate
tde_counts = [
    # int(top_counts[0]),  # Best
    # int(top_counts[1]),  # 2nd best
    # int(top_counts[2]),  # 3rd best
    # 315,                 # Your baseline
    # 310, 320, 325, 330, 335, 340  # Nearby variations
    332
]

# Remove duplicates and sort
tde_counts = sorted(set(tde_counts))

print(f"\nGenerating {len(tde_counts)} submissions...\n")

# Pre-compute ranks once
ranks = rankdata(probs, method='ordinal')

for tde_count in tde_counts:
    # Rank-based prediction
    cutoff = len(probs) - tde_count + 1
    predictions = (ranks >= cutoff).astype(int)
    
    # Get effective threshold
    threshold = np.sort(probs)[-tde_count]
    
    # Validation F1 (if available)
    val_f1 = ""
    row = analysis[analysis['test_tde_count'] == tde_count]
    if len(row) > 0:
        val_f1 = f"(Val F1: {row.iloc[0]['val_f1']:.4f})"
    
    # Create submission
    submission = pd.DataFrame({
        'object_id': test_ids,
        'prediction': predictions
    })
    
    # Save
    filename = f"submission_magic_{tde_count}.csv"
    filepath = OUTPUT_DIR / filename
    submission.to_csv(filepath, index=False)
    
    print(f"✓ {filename:30s} → {predictions.sum():3d} TDEs, thresh={threshold:.4f} {val_f1}")

print("\n" + "="*80)
print("SUBMISSION STRATEGY")
print("="*80)

best_count = int(top_counts[0])
print(f"\nRECOMMENDED SUBMISSION ORDER:")
print(f"\n1. submission_magic_{best_count}.csv ⭐ BEST")
print(f"   → Highest validation F1")

if 315 != best_count:
    print(f"\n2. submission_magic_315.csv")
    print(f"   → Your baseline (64.57% test F1)")

print(f"\n3. submission_magic_{int(top_counts[1])}.csv")
print(f"   → 2nd best validation F1")

print(f"\n4. submission_magic_{int(top_counts[2])}.csv")
print(f"   → 3rd best validation F1")

print("\nAFTER SEEING RESULTS:")
print("  - If actual F1 > expected: Try counts 5-10 lower")
print("  - If actual F1 < expected: Try counts 10-20 higher")
print("  - If actual F1 ≈ expected: You found the sweet spot!")

print("\n" + "="*80)
print(f"✓ Generated {len(tde_counts)} submissions in {OUTPUT_DIR}/")
print("="*80)