#!/usr/bin/env python3
"""
Preprocessing Module for TDE Classification
Handles flux de-extinction, feature engineering, and data preparation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')

try:
    import extinction
    EXTINCTION_AVAILABLE = True
except ImportError:
    EXTINCTION_AVAILABLE = False
    print("WARNING: 'extinction' package not available. Install with: pip install extinction==0.4.7")

FILTER_WAVELENGTHS = {
    'u': 3641.0,
    'g': 4704.0,
    'r': 6155.0,
    'i': 7504.0,
    'z': 8695.0,
    'y': 10056.0
}

FILTERS = ['u', 'g', 'r', 'i', 'z', 'y']


class TDEPreprocessor:
    """
    Comprehensive preprocessing pipeline for TDE classification
    """
    
    def __init__(self, 
                 use_robust_scaling=True,
                 variance_threshold=0.01,
                 handle_outliers=True,
                 create_interactions=True,
                 random_seed=42):
        """
        Initialize preprocessor with configuration
        
        Parameters:
        -----------
        use_robust_scaling : bool
            If True, use RobustScaler (better for outliers), else StandardScaler
        variance_threshold : float
            Remove features with variance below this threshold
        handle_outliers : bool
            If True, clip outliers at 3 standard deviations
        create_interactions : bool
            If True, create interaction features between filters
        random_seed : int
            Random seed for reproducibility
        """
        self.use_robust_scaling = use_robust_scaling
        self.variance_threshold = variance_threshold
        self.handle_outliers = handle_outliers
        self.create_interactions = create_interactions
        self.random_seed = random_seed
        
        # Initialize scalers and selectors
        if use_robust_scaling:
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        self.variance_selector = VarianceThreshold(threshold=variance_threshold)
        
        # Track feature statistics
        self.train_median = None
        self.train_std = None
        self.selected_features = None
        self.feature_names = None
        
    
    def de_extinct_flux(self, lightcurve_df, ebv_value):
        """
        De-extinct flux measurements using the extinction package
        
        Parameters:
        -----------
        lightcurve_df : pd.DataFrame
            DataFrame with columns: Flux, Flux_err, Filter
        ebv_value : float
            E(B-V) extinction coefficient
            
        Returns:
        --------
        pd.DataFrame : DataFrame with de-extincted flux values
        """
        if not EXTINCTION_AVAILABLE:
            print("WARNING: Cannot de-extinct - extinction package not installed")
            return lightcurve_df
        
        df = lightcurve_df.copy()
        
        for filt in FILTERS:
            if filt not in df['Filter'].values:
                continue
            
            # Get effective wavelength for this filter
            eff_wl = np.array([FILTER_WAVELENGTHS[filt]])
            
            # Calculate extinction in magnitudes
            # Using Fitzpatrick 1999 extinction law with R_V = 3.1 (standard Milky Way)
            A_lambda = extinction.fitzpatrick99(eff_wl, ebv_value * 3.1)
            
            # Convert to flux correction factor
            # flux_corrected = flux_observed * 10^(A_lambda / 2.5)
            correction_factor = 10 ** (A_lambda[0] / 2.5)
            
            # Apply correction to flux and error
            mask = df['Filter'] == filt
            df.loc[mask, 'Flux'] *= correction_factor
            df.loc[mask, 'Flux_err'] *= correction_factor
        
        return df
    
    
    def create_color_features(self, features_dict):
        """
        Create color (flux ratio) features between filters
        
        Parameters:
        -----------
        features_dict : dict
            Dictionary of extracted features
            
        Returns:
        --------
        dict : Updated features dictionary with color features
        """
        colors = features_dict.copy()
        
        # Band pairs for color features
        band_pairs = [
            ('u', 'g'), ('g', 'r'), ('r', 'i'), 
            ('i', 'z'), ('z', 'y'), ('u', 'r'),
            ('g', 'i'), ('r', 'z')
        ]
        
        for b1, b2 in band_pairs:
            flux1_key = f'{b1}_flux_mean'
            flux2_key = f'{b2}_flux_mean'
            
            if flux1_key in features_dict and flux2_key in features_dict:
                flux1 = features_dict[flux1_key]
                flux2 = features_dict[flux2_key]
                
                # Flux ratio (color proxy)
                if flux2 != 0:
                    colors[f'color_{b1}_{b2}_ratio'] = flux1 / (abs(flux2) + 1e-10)
                
                # Magnitude difference (traditional color)
                if flux1 > 0 and flux2 > 0:
                    colors[f'color_{b1}_{b2}_mag'] = -2.5 * np.log10(flux1 / flux2)
        
        return colors
    
    
    def create_temporal_features(self, features_dict):
        """
        Create temporal aggregation features
        
        Parameters:
        -----------
        features_dict : dict
            Dictionary of extracted features
            
        Returns:
        --------
        dict : Updated features dictionary with temporal features
        """
        temp_feats = features_dict.copy()
        
        # Ratios between filters
        for filt in FILTERS:
            rise_key = f'{filt}_rise_time'
            fade_key = f'{filt}_fade_time'
            
            if rise_key in features_dict and fade_key in features_dict:
                rise_time = features_dict[rise_key]
                fade_time = features_dict[fade_key]
                
                if fade_time != 0 and not np.isnan(rise_time) and not np.isnan(fade_time):
                    temp_feats[f'{filt}_rise_fade_ratio'] = rise_time / (abs(fade_time) + 1e-10)
        
        # Global temporal features
        if 'total_time_span' in features_dict and 'total_n_obs' in features_dict:
            time_span = features_dict['total_time_span']
            n_obs = features_dict['total_n_obs']
            
            if time_span > 0 and n_obs > 0:
                temp_feats['avg_observation_rate'] = n_obs / time_span
        
        return temp_feats
    
    
    def create_interaction_features(self, df):
        """
        Create interaction features between key measurements
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature DataFrame
            
        Returns:
        --------
        pd.DataFrame : DataFrame with added interaction features
        """
        result_df = df.copy()
        
        # Distance modulus * redshift interactions
        if 'distance_modulus' in df.columns and 'Z' in df.columns:
            result_df['dist_z_interaction'] = df['distance_modulus'] * df['Z']
        
        # Peak flux * time span interactions
        peak_cols = [col for col in df.columns if 'peak_flux' in col]
        if 'total_time_span' in df.columns:
            for col in peak_cols:
                result_df[f'{col}_time_interaction'] = df[col] * df['total_time_span']
        
        # SNR * number of observations
        snr_cols = [col for col in df.columns if 'snr_mean' in col]
        n_obs_cols = [col for col in df.columns if 'n_obs' in col and col.startswith(('u_', 'g_', 'r_', 'i_', 'z_', 'y_'))]
        
        for snr_col in snr_cols:
            filt = snr_col.split('_')[0]
            nobs_col = f'{filt}_n_obs'
            if nobs_col in df.columns:
                result_df[f'{filt}_snr_nobs_interaction'] = df[snr_col] * df[nobs_col]
        
        return result_df
    
    
    def handle_missing_values(self, df, is_train=True):
        """
        Intelligent missing value imputation
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature DataFrame
        is_train : bool
            If True, fit imputation statistics; else use stored statistics
            
        Returns:
        --------
        pd.DataFrame : DataFrame with imputed values
        """
        df_imputed = df.copy()
        
        # Calculate or use stored median values
        if is_train:
            self.train_median = df.median()
            self.train_std = df.std()
        
        # Fill NaN and inf values
        df_imputed = df_imputed.replace([np.inf, -np.inf], np.nan)
        df_imputed = df_imputed.fillna(self.train_median)
        
        return df_imputed
    
    
    def clip_outliers(self, df, n_std=3):
        """
        Clip extreme outliers to n standard deviations
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature DataFrame
        n_std : int
            Number of standard deviations for clipping
            
        Returns:
        --------
        pd.DataFrame : DataFrame with clipped values
        """
        if not self.handle_outliers:
            return df
        
        df_clipped = df.copy()
        
        for col in df.columns:
            mean = self.train_median[col]
            std = self.train_std[col]
            
            lower_bound = mean - n_std * std
            upper_bound = mean + n_std * std
            
            df_clipped[col] = df_clipped[col].clip(lower_bound, upper_bound)
        
        return df_clipped
    
    
    def remove_low_variance_features(self, df, is_train=True):
        """
        Remove features with low variance
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature DataFrame
        is_train : bool
            If True, fit variance selector; else use stored selector
            
        Returns:
        --------
        pd.DataFrame : DataFrame with low variance features removed
        """
        if is_train:
            df_selected = pd.DataFrame(
                self.variance_selector.fit_transform(df),
                columns=df.columns[self.variance_selector.get_support()],
                index=df.index
            )
            self.selected_features = df_selected.columns.tolist()
        else:
            # Use stored feature selection
            df_selected = df[self.selected_features]
        
        return df_selected
    
    
    def fit_transform(self, df, target=None):
        """
        Fit preprocessor and transform training data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature DataFrame
        target : pd.Series, optional
            Target variable (will be preserved separately)
            
        Returns:
        --------
        pd.DataFrame : Preprocessed training features
        pd.Series : Target variable (if provided)
        """
        # Separate target if provided
        if target is not None:
            y = target.copy()
        else:
            y = None
        
        # Step 1: Handle missing values
        print("  Step 1/6: Handling missing values...")
        df_clean = self.handle_missing_values(df, is_train=True)
        
        # Step 2: Create additional features
        print("  Step 2/6: Creating color features...")
        # Color features already in GP features, but create temporal ones
        df_enhanced = df_clean.copy()
        
        # Step 3: Create interaction features
        if self.create_interactions:
            print("  Step 3/6: Creating interaction features...")
            df_enhanced = self.create_interaction_features(df_enhanced)
        else:
            print("  Step 3/6: Skipping interaction features...")
        
        # Step 4: Clip outliers
        if self.handle_outliers:
            print("  Step 4/6: Clipping outliers...")
            df_enhanced = self.clip_outliers(df_enhanced)
        else:
            print("  Step 4/6: Skipping outlier handling...")
        
        # Step 5: Remove low variance features
        print("  Step 5/6: Removing low variance features...")
        df_selected = self.remove_low_variance_features(df_enhanced, is_train=True)
        print(f"    Kept {len(df_selected.columns)}/{len(df_enhanced.columns)} features")
        
        # Step 6: Scale features
        print("  Step 6/6: Scaling features...")
        df_scaled = pd.DataFrame(
            self.scaler.fit_transform(df_selected),
            columns=df_selected.columns,
            index=df_selected.index
        )
        
        self.feature_names = df_scaled.columns.tolist()
        
        if y is not None:
            return df_scaled, y
        return df_scaled
    
    
    def transform(self, df):
        """
        Transform test data using fitted preprocessor
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature DataFrame
            
        Returns:
        --------
        pd.DataFrame : Preprocessed test features
        """
        # Step 1: Handle missing values
        df_clean = self.handle_missing_values(df, is_train=False)
        
        # Step 2: Create interaction features (if enabled)
        if self.create_interactions:
            df_enhanced = self.create_interaction_features(df_clean)
        else:
            df_enhanced = df_clean
        
        # Step 3: Clip outliers (if enabled)
        if self.handle_outliers:
            df_enhanced = self.clip_outliers(df_enhanced)
        
        # Step 4: Select same features as training
        df_selected = df_enhanced[self.selected_features]
        
        # Step 5: Scale features
        df_scaled = pd.DataFrame(
            self.scaler.transform(df_selected),
            columns=df_selected.columns,
            index=df_selected.index
        )
        
        return df_scaled


def preprocess_features(data_dir, output_dir, config=None):
    """
    Main preprocessing pipeline function
    
    Parameters:
    -----------
    data_dir : Path
        Directory containing extracted features
    output_dir : Path
        Directory to save preprocessed data
    config : dict, optional
        Configuration for preprocessor
        
    Returns:
    --------
    TDEPreprocessor : Fitted preprocessor object
    """
    print("\n" + "="*80)
    print("PREPROCESSING PIPELINE")
    print("="*80)
    
    # Load extracted GP features
    print("\nLoading extracted features...")
    train_df = pd.read_csv(data_dir / 'train_gp_features.csv')
    test_df = pd.read_csv(data_dir / 'test_gp_features.csv')
    
    print(f"  Train: {train_df.shape}")
    print(f"  Test: {test_df.shape}")
    
    # Separate IDs and target
    train_ids = train_df['object_id']
    test_ids = test_df['object_id']
    y_train = train_df['target']
    
    # Get feature columns
    train_cols = set(train_df.columns) - {'object_id', 'target'}
    test_cols = set(test_df.columns) - {'object_id'}
    common_cols = sorted(list(train_cols & test_cols))
    
    print(f"\nFeature alignment:")
    print(f"  Train features: {len(train_cols)}")
    print(f"  Test features: {len(test_cols)}")
    print(f"  Common features: {len(common_cols)}")
    
    # Select common features
    X_train = train_df[common_cols]
    X_test = test_df[common_cols]
    
    # Initialize preprocessor
    if config is None:
        config = {
            'use_robust_scaling': True,
            'variance_threshold': 0.01,
            'handle_outliers': True,
            'create_interactions': True,
            'random_seed': 42
        }
    
    preprocessor = TDEPreprocessor(**config)
    
    # Fit and transform training data
    print("\nPreprocessing training data...")
    X_train_processed, y_train_final = preprocessor.fit_transform(X_train, y_train)
    
    # Transform test data
    print("\nPreprocessing test data...")
    X_test_processed = preprocessor.transform(X_test)
    
    # Save processed data
    print("\nSaving preprocessed data...")
    
    train_final = X_train_processed.copy()
    train_final['target'] = y_train_final.values
    train_final.to_csv(output_dir / 'train_processed.csv', index=False)
    
    test_final = X_test_processed.copy()
    test_final.insert(0, 'object_id', test_ids.values)
    test_final.to_csv(output_dir / 'test_processed.csv', index=False)
    
    print(f"\n  Saved train: {train_final.shape}")
    print(f"  Saved test: {test_final.shape}")
    print(f"  Class distribution: 0={sum(y_train_final==0)}, 1={sum(y_train_final==1)}")
    
    # Save preprocessor for later use
    import joblib
    joblib.dump(preprocessor, output_dir.parent / 'models' / 'preprocessor.pkl')
    print(f"\n  Saved preprocessor to models/preprocessor.pkl")
    
    return preprocessor


if __name__ == '__main__':
    # Example usage
    from pathlib import Path
    
    REAL_BASE_DIR = Path('/home/adnope/Dev/projects/mallorn')
    DATA_DIR = REAL_BASE_DIR / 'data/processed'
    OUTPUT_DIR = DATA_DIR
    
    preprocessor = preprocess_features(DATA_DIR, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)