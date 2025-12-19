#!/usr/bin/env python3
"""
Approach 5: Gaussian Process Light Curve Modeling (Updated)
Based on PLAsTiCC winning strategy - GP features + smart preprocessing + XGBoost
NOW WITH: Modular preprocessing including flux de-extinction
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import our new preprocessing module
from preprocessing import TDEPreprocessor, preprocess_features, EXTINCTION_AVAILABLE
if EXTINCTION_AVAILABLE:
    import extinction

BASE_DIR = Path('/home/adnope/Dev/projects/mallorn/data/raw')
REAL_BASE_DIR = Path('/home/adnope/Dev/projects/mallorn')
DATA_DIR = REAL_BASE_DIR / 'data/processed'
MODEL_DIR = REAL_BASE_DIR / 'models'
OUTPUT_DIR = REAL_BASE_DIR / 'outputs'

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FILTERS = ['u', 'g', 'r', 'i', 'z', 'y']

# ==================== GP FEATURE EXTRACTION (WITH DE-EXTINCTION) ====================

def fit_gp_to_lightcurve(times, flux, flux_err):
    """Fit Gaussian Process to a single-band lightcurve"""
    if len(times) < 3:
        return None, None
    
    times = times - times.min()
    kernel = 1.0 * Matern(length_scale=50.0, nu=2.5) + WhiteKernel(noise_level=0.1)
    
    try:
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=flux_err**2 + 1e-6,
            n_restarts_optimizer=2,
            random_state=RANDOM_SEED
        )
        gp.fit(times.reshape(-1, 1), flux)
        return gp, times
    except:
        return None, None


def extract_gp_features(times, flux, flux_err, filter_name):
    """Extract GP-based features from a single-band lightcurve"""
    features = {}
    prefix = f'{filter_name}_'
    
    if len(times) < 3:
        return features
    
    # Basic statistics
    features[f'{prefix}n_obs'] = len(times)
    features[f'{prefix}flux_mean'] = np.mean(flux)
    features[f'{prefix}flux_std'] = np.std(flux)
    features[f'{prefix}flux_max'] = np.max(flux)
    features[f'{prefix}flux_min'] = np.min(flux)
    features[f'{prefix}flux_range'] = np.max(flux) - np.min(flux)
    features[f'{prefix}flux_skew'] = stats.skew(flux)
    features[f'{prefix}flux_kurtosis'] = stats.kurtosis(flux)
    
    # SNR features
    snr = flux / (flux_err + 1e-10)
    features[f'{prefix}snr_mean'] = np.mean(snr)
    features[f'{prefix}snr_max'] = np.max(snr)
    
    # Time-based features
    times_norm = times - times.min()
    features[f'{prefix}time_span'] = times_norm.max()
    
    if len(times) > 1:
        time_diffs = np.diff(np.sort(times))
        features[f'{prefix}cadence_mean'] = np.mean(time_diffs)
        features[f'{prefix}cadence_std'] = np.std(time_diffs)
    
    # Peak features
    peak_idx = np.argmax(flux)
    features[f'{prefix}peak_flux'] = flux[peak_idx]
    features[f'{prefix}peak_time'] = times_norm[peak_idx]
    
    # Fit GP and extract kernel parameters
    gp, times_gp = fit_gp_to_lightcurve(times, flux, flux_err)
    
    if gp is not None:
        try:
            params = gp.kernel_.get_params()
            for key, val in params.items():
                if isinstance(val, (int, float)) and not np.isnan(val) and not np.isinf(val):
                    features[f'{prefix}gp_{key}'] = val
            
            # GP predictions
            t_pred = np.linspace(0, times_gp.max(), 50).reshape(-1, 1)
            flux_pred, flux_std = gp.predict(t_pred, return_std=True)
            
            # Rise and fade times
            peak_val = np.max(flux_pred)
            peak_idx_pred = np.argmax(flux_pred)
            half_peak = peak_val / 2
            
            # Rise time
            pre_peak = flux_pred[:peak_idx_pred+1]
            if len(pre_peak) > 1 and np.any(pre_peak < half_peak):
                rise_idx = np.where(pre_peak < half_peak)[0][-1]
                features[f'{prefix}rise_time'] = t_pred[peak_idx_pred, 0] - t_pred[rise_idx, 0]
            
            # Fade time
            post_peak = flux_pred[peak_idx_pred:]
            if len(post_peak) > 1 and np.any(post_peak < half_peak):
                fade_idx = np.where(post_peak < half_peak)[0][0] + peak_idx_pred
                features[f'{prefix}fade_time'] = t_pred[fade_idx, 0] - t_pred[peak_idx_pred, 0]
            
            # GP smoothness
            features[f'{prefix}gp_smoothness'] = np.mean(flux_std)
            
            # Rate of change
            flux_deriv = np.diff(flux_pred) / np.diff(t_pred.flatten())
            features[f'{prefix}max_rise_rate'] = np.max(flux_deriv)
            features[f'{prefix}max_fade_rate'] = np.min(flux_deriv)
        except:
            pass
    
    return features


def extract_color_features(lc_data):
    """Extract color evolution features"""
    features = {}
    
    band_flux = {}
    for filt in FILTERS:
        filt_data = lc_data[lc_data['Filter'] == filt]
        if len(filt_data) > 0:
            band_flux[filt] = np.mean(filt_data['Flux'].values)
    
    color_pairs = [('u', 'g'), ('g', 'r'), ('r', 'i'), ('i', 'z'), ('z', 'y')]
    
    for b1, b2 in color_pairs:
        if b1 in band_flux and b2 in band_flux:
            if band_flux[b2] != 0:
                features[f'color_{b1}_{b2}'] = band_flux[b1] / (band_flux[b2] + 1e-10)
            
            if band_flux[b1] > 0 and band_flux[b2] > 0:
                features[f'mag_diff_{b1}_{b2}'] = -2.5 * np.log10(band_flux[b1] / band_flux[b2])
    
    # Blue excess
    if 'u' in band_flux and 'r' in band_flux:
        if band_flux['u'] > 0 and band_flux['r'] > 0:
            features['blue_excess'] = -2.5 * np.log10(band_flux['u'] / band_flux['r'])
    
    return features


def power_law_decay(t, A, alpha, t0):
    """Power law decay model for TDEs"""
    return A * np.power(np.maximum(t - t0, 0.01), -alpha)


def fit_power_law_features(times, flux):
    """Fit power law decay"""
    features = {}
    
    if len(times) < 5:
        return features
    
    times_norm = times - times.min()
    peak_idx = np.argmax(flux)
    
    post_peak_mask = np.arange(len(times)) >= peak_idx
    t_post = times_norm[post_peak_mask]
    f_post = flux[post_peak_mask]
    
    if len(t_post) >= 3:
        try:
            popt, pcov = curve_fit(
                power_law_decay,
                t_post, f_post,
                p0=[np.max(f_post), 1.0, -10],
                bounds=([0, 0.1, -100], [np.inf, 5.0, 100]),
                maxfev=1000
            )
            
            features['powerlaw_A'] = popt[0]
            features['powerlaw_alpha'] = popt[1]
            features['powerlaw_t0'] = popt[2]
            
            y_pred = power_law_decay(t_post, *popt)
            residuals = f_post - y_pred
            features['powerlaw_chi2'] = np.sum(residuals**2) / (len(f_post) - 3)
        except:
            pass
    
    return features


def extract_all_features(obj_id, lc_data, meta_row, preprocessor=None):
    """
    Extract all features for a single object
    NOW WITH: Flux de-extinction applied first
    """
    features = {'object_id': obj_id}
    
    # Metadata features
    features['Z'] = meta_row['Z']
    features['EBV'] = meta_row['EBV']
    
    # Distance modulus from redshift
    if meta_row['Z'] > 0:
        features['distance_modulus'] = 5 * np.log10(meta_row['Z'] * 3e5 / 70) + 25
    
    # DE-EXTINCT FLUX (NEW!)
    if preprocessor is not None and EXTINCTION_AVAILABLE:
        lc_data = preprocessor.de_extinct_flux(lc_data, meta_row['EBV'])
    
    # Per-band GP features
    all_flux = []
    all_times = []
    
    for filt in FILTERS:
        filt_data = lc_data[lc_data['Filter'] == filt]
        if len(filt_data) > 0:
            times = filt_data['Time (MJD)'].values
            flux = filt_data['Flux'].values
            flux_err = filt_data['Flux_err'].values
            
            gp_feats = extract_gp_features(times, flux, flux_err, filt)
            features.update(gp_feats)
            
            all_flux.extend(flux)
            all_times.extend(times)
    
    # Color features
    color_feats = extract_color_features(lc_data)
    features.update(color_feats)
    
    # Power law fit
    if len(all_flux) > 0:
        all_flux = np.array(all_flux)
        all_times = np.array(all_times)
        
        sort_idx = np.argsort(all_times)
        all_times = all_times[sort_idx]
        all_flux = all_flux[sort_idx]
        
        pl_feats = fit_power_law_features(all_times, all_flux)
        features.update(pl_feats)
    
    # Global features
    features['total_n_obs'] = len(all_flux)
    features['total_flux_mean'] = np.mean(all_flux) if len(all_flux) > 0 else 0
    features['total_flux_max'] = np.max(all_flux) if len(all_flux) > 0 else 0
    features['total_time_span'] = all_times.max() - all_times.min() if len(all_times) > 0 else 0
    
    return features


# ==================== DATA LOADING ====================

def load_all_lightcurves(mode='train'):
    """Load all lightcurves from all splits"""
    all_lc = []
    for split_num in range(1, 21):
        split_dir = BASE_DIR / f'split_{split_num:02d}'
        lc_file = split_dir / f'{mode}_full_lightcurves.csv'
        
        if lc_file.exists():
            lc_split = pd.read_csv(lc_file)
            all_lc.append(lc_split)
    
    if all_lc:
        return pd.concat(all_lc, ignore_index=True)
    return pd.DataFrame()


def step1_extract_gp_features():
    """Extract GP-based features from lightcurves WITH de-extinction"""
    print("\n" + "="*80)
    print("STEP 1: Gaussian Process Feature Extraction (WITH DE-EXTINCTION)")
    print("="*80)
    
    # Initialize preprocessor for de-extinction
    preprocessor = TDEPreprocessor() if EXTINCTION_AVAILABLE else None
    
    for mode in ['train', 'test']:
        print(f"\nProcessing {mode}...")
        
        # Load metadata
        meta_file = 'train_log.csv' if mode == 'train' else 'test_log.csv'
        meta_df = pd.read_csv(BASE_DIR / meta_file)
        print(f"  Loaded {len(meta_df)} objects metadata")
        
        # Load lightcurves
        lc_df = load_all_lightcurves(mode)
        print(f"  Loaded {len(lc_df)} lightcurve points")
        
        # Extract features
        all_features = []
        for idx, row in meta_df.iterrows():
            obj_id = row['object_id']
            obj_lc = lc_df[lc_df['object_id'] == obj_id]
            
            if len(obj_lc) > 0:
                features = extract_all_features(obj_id, obj_lc, row, preprocessor)
                if mode == 'train':
                    features['target'] = row['target']
                all_features.append(features)
            
            if (idx + 1) % 200 == 0:
                print(f"  Processed {idx + 1}/{len(meta_df)} objects...")
        
        result_df = pd.DataFrame(all_features)
        result_df.to_csv(DATA_DIR / f'{mode}_gp_features.csv', index=False)
        print(f"  Extracted {len(result_df)} objects with {len(result_df.columns)} features")


# ==================== PREPROCESSING (NOW MODULAR!) ====================

def step2_preprocess():
    """
    Preprocess features using the new modular preprocessing pipeline
    """
    print("\n" + "="*80)
    print("STEP 2: Preprocessing (MODULAR)")
    print("="*80)
    
    # Configuration for preprocessing
    config = {
        'use_robust_scaling': True,  # Better for outliers
        'variance_threshold': 0.01,  # Remove very low variance features
        'handle_outliers': True,      # Clip outliers at 3 std
        'create_interactions': True,  # Create interaction features
        'random_seed': RANDOM_SEED
    }
    
    # Run preprocessing pipeline
    preprocessor = preprocess_features(DATA_DIR, DATA_DIR, config)
    
    return preprocessor


# ==================== MODEL TRAINING ====================

def step3_train():
    """Train ensemble of models with stratified K-fold"""
    print("\n" + "="*80)
    print("STEP 3: Training Ensemble Models")
    print("="*80)
    
    train_df = pd.read_csv(DATA_DIR / 'train_processed.csv')
    X = train_df.drop('target', axis=1).values
    y = train_df['target'].values
    feature_names = train_df.drop('target', axis=1).columns.tolist()
    
    print(f"Training data: {X.shape}")
    print(f"Class balance: {sum(y==0)} / {sum(y==1)} = {sum(y==0)/sum(y==1):.1f}:1")
    
    scale_pos_weight = sum(y == 0) / sum(y == 1)
    
    # Stratified K-Fold
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    oof_preds = np.zeros(len(y))
    models = {'xgb': [], 'lgb': [], 'cat': []}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        fold_preds = np.zeros(len(val_idx))
        
        # XGBoost
        print("  Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            random_state=RANDOM_SEED,
            early_stopping_rounds=50,
            verbosity=0
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
        fold_preds += xgb_pred / 3
        models['xgb'].append(xgb_model)
        print(f"  XGB F1: {f1_score(y_val, (xgb_pred > 0.5).astype(int)):.4f}")
        
        # LightGBM
        print("  Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            num_leaves=31,
            class_weight='balanced',
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=RANDOM_SEED,
            verbose=-1
        )
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
        fold_preds += lgb_pred / 3
        models['lgb'].append(lgb_model)
        print(f"  LGB F1: {f1_score(y_val, (lgb_pred > 0.5).astype(int)):.4f}")
        
        # CatBoost
        print("  Training CatBoost...")
        cat_model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.03,
            auto_class_weights='Balanced',
            random_seed=RANDOM_SEED,
            verbose=False,
            early_stopping_rounds=50
        )
        cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        cat_pred = cat_model.predict_proba(X_val)[:, 1]
        fold_preds += cat_pred / 3
        models['cat'].append(cat_model)
        print(f"  CAT F1: {f1_score(y_val, (cat_pred > 0.5).astype(int)):.4f}")
        
        oof_preds[val_idx] = fold_preds
        
        ensemble_f1 = f1_score(y_val, (fold_preds > 0.5).astype(int))
        print(f"  Ensemble F1: {ensemble_f1:.4f}")
    
    # Save models
    joblib.dump(models, MODEL_DIR / 'ensemble_models.pkl')
    
    # Find optimal threshold
    print("\n" + "="*80)
    print("Finding Optimal Threshold")
    print("="*80)
    
    best_threshold = 0.5
    best_f1 = 0
    
    for thresh in np.linspace(0.1, 0.9, 81):
        f1 = f1_score(y, (oof_preds > thresh).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    print(f"Optimal threshold: {best_threshold:.4f}")
    print(f"Best OOF F1: {best_f1:.4f}")
    print(f"ROC-AUC: {roc_auc_score(y, oof_preds):.4f}")
    
    # Confusion matrix
    y_pred_opt = (oof_preds > best_threshold).astype(int)
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred_opt))
    
    # Save threshold
    pd.DataFrame({'optimal_threshold': [best_threshold]}).to_csv(
        OUTPUT_DIR / 'optimal_threshold_gp.csv', index=False
    )
    
    # Feature importance
    importance = np.mean([m.feature_importances_ for m in models['xgb']], axis=0)
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Features:")
    print(feat_imp.head(20).to_string(index=False))
    
    feat_imp.to_csv(OUTPUT_DIR / 'feature_importance_gp.csv', index=False)


# ==================== PREDICTION ====================

def step4_predict():
    """Make predictions using ensemble"""
    print("\n" + "="*80)
    print("STEP 4: Making Predictions")
    print("="*80)
    
    test_df = pd.read_csv(DATA_DIR / 'test_processed.csv')
    test_ids = test_df['object_id'].values
    X_test = test_df.drop('object_id', axis=1).values
    
    print(f"Test data: {X_test.shape}")
    
    # Load models
    models = joblib.load(MODEL_DIR / 'ensemble_models.pkl')
    threshold = pd.read_csv(OUTPUT_DIR / 'optimal_threshold_gp.csv')['optimal_threshold'].values[0]
    
    print(f"Using threshold: {threshold:.4f}")
    
    # Ensemble predictions
    test_preds = np.zeros(len(X_test))
    
    for model_type, model_list in models.items():
        for model in model_list:
            test_preds += model.predict_proba(X_test)[:, 1]
    
    test_preds /= (len(models['xgb']) + len(models['lgb']) + len(models['cat']))
    
    # Apply threshold
    predictions = (test_preds > threshold).astype(int)
    
    # Create submission
    submission = pd.DataFrame({
        'object_id': test_ids,
        'prediction': predictions
    })
    
    submission.to_csv(OUTPUT_DIR / 'submission_gp_ensemble.csv', index=False)
    
    print(f"\nPredictions:")
    print(f"  Class 0: {sum(predictions == 0)}")
    print(f"  Class 1: {sum(predictions == 1)}")
    print(f"  Detection rate: {sum(predictions == 1) / len(predictions) * 100:.2f}%")
    print(f"\nSubmission saved: {OUTPUT_DIR / 'submission_gp_ensemble.csv'}")


# ==================== MAIN ====================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("GAUSSIAN PROCESS ENSEMBLE PIPELINE (IMPROVED)")
    print("Based on PLAsTiCC + Advanced Preprocessing")
    print("="*80)
    
    step1_extract_gp_features()
    step2_preprocess()
    step3_train()
    step4_predict()
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)