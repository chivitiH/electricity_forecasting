"""
Electricity Consumption Forecasting - Data Preprocessing Pipeline
Fichier: src/data_preprocessing.py
Adapté pour dataset PJM réel (PJME_hourly.csv avec colonnes Datetime, PJME_MW)
"""
import os
os.environ['SKLEARNEX_VERBOSE'] = 'INFO'
from sklearnex import patch_sklearn
patch_sklearn()
import warnings


# Intel optimizations for NumPy/Pandas
try:
    import mkl
    print(f"✅ Intel MKL configuré: {mkl.get_max_threads()} threads")
except ImportError:
    print("⚠️ Intel MKL non disponible")
    
import pandas as pd
import numpy as np
import pickle
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path
from scipy import stats
import holidays

from config import config, BUSINESS_METRICS, PJM_BUSINESS_RULES

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class PJMPreprocessor:
    """
    Preprocessing pipeline pour données PJM réelles
    - Charge PJME_hourly.csv (145k points, 2002-2018)
    - Feature engineering électricité (patterns horaires/saisonniers)
    - Gestion multi-régions PJM
    - Préparation pour SARIMA forecasting
    """
    
    def __init__(self, region: str = "PJME"):
        self.config = config
        self.region = region
        
        # Configure pour la région sélectionnée
        self.config.switch_pjm_region(region)
        
        # Storage datasets
        self.consumption_data = None
        self.weather_data = None
        self.holidays_data = None
        
        logger.info(f"PJMPreprocessor initialisé pour région: {region}")
        logger.info(f"Fichier: {self.config.data.consumption_file}")
        logger.info(f"Colonne: {self.config.data.consumption_col}")
    
    def load_pjm_data(self) -> None:
        """Chargement données PJM depuis Kaggle"""
        logger.info(f"📊 Chargement données PJM {self.region}...")
        
        try:
            # Dataset principal PJM
            consumption_path = Path(self.config.data.raw_data_path) / self.config.data.consumption_file
            
            if not consumption_path.exists():
                raise FileNotFoundError(f"Fichier {consumption_path} non trouvé")
            
            self.consumption_data = pd.read_csv(consumption_path)
            
            # Validation structure dataset PJM
            expected_cols = [self.config.data.datetime_col, self.config.data.consumption_col]
            if not all(col in self.consumption_data.columns for col in expected_cols):
                logger.error(f"❌ Colonnes attendues: {expected_cols}")
                logger.error(f"❌ Colonnes trouvées: {self.consumption_data.columns.tolist()}")
                raise ValueError("Structure dataset PJM incorrecte")
            
            # Basic info
            logger.info(f"✅ Dataset {self.region} chargé:")
            logger.info(f"  Shape: {self.consumption_data.shape}")
            logger.info(f"  Période: {self.consumption_data[self.config.data.datetime_col].min()} → {self.consumption_data[self.config.data.datetime_col].max()}")
            logger.info(f"  Consommation moyenne: {self.consumption_data[self.config.data.consumption_col].mean():.1f} MW")
            
            # Génération données auxiliaires
            self._generate_weather_data()
            self._generate_holidays_data()
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement {self.region}: {e}")
            logger.info("💡 Vérifiez que les fichiers CSV PJM sont dans data/raw/")
            raise
    
    def _generate_weather_data(self) -> None:
        """Génération données météo synthétiques réalistes"""
        logger.info("🌤️  Génération données météo synthétiques...")
        
        if self.consumption_data is None:
            return
        
        # Dates du dataset PJM
        dates = pd.to_datetime(self.consumption_data[self.config.data.datetime_col])
        n_hours = len(dates)
        
        # Température avec saisonnalité réaliste pour région PJM (Est USA)
        day_of_year = dates.dt.dayofyear
        hour_of_day = dates.dt.hour
        
        # Cycle annuel (base: 10°C hiver, 25°C été)
        annual_temp = 17.5 + 12.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Cycle quotidien (variation ±4°C)
        daily_temp_variation = 4 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        
        # Bruit réaliste
        noise = np.random.normal(0, 3, n_hours)
        
        # Corrélation inverse consommation-température (plus froid/chaud = plus consommation)
        consumption_values = self.consumption_data[self.config.data.consumption_col].values
        consumption_normalized = (consumption_values - consumption_values.mean()) / consumption_values.std()
        
        # Ajustement température basé sur consommation
        temp_adjustment = -2 * consumption_normalized  # Inverse correlation
        
        final_temperature = annual_temp + daily_temp_variation + noise + temp_adjustment
        
        self.weather_data = pd.DataFrame({
            self.config.data.datetime_col: dates,
            self.config.data.temperature_col: final_temperature,
            'humidity': 65 + 25 * np.random.random(n_hours),  # 40-90%
            'pressure': 1013 + 20 * np.random.randn(n_hours),  # ±20 hPa
            'wind_speed': np.abs(np.random.normal(8, 4, n_hours))  # km/h
        })
        
        logger.info(f"✅ Météo synthétique créée: {self.weather_data.shape}")
        logger.info(f"  Température: {final_temperature.min():.1f}°C → {final_temperature.max():.1f}°C")
    
    def _generate_holidays_data(self) -> None:
        """Génération jours fériés USA (région PJM) - Version corrigée"""
        logger.info("📅 Génération jours fériés USA...")
        
        if self.consumption_data is None:
            return
        
        dates = pd.to_datetime(self.consumption_data[self.config.data.datetime_col])
        start_year = dates.min().year
        end_year = dates.max().year
        
        # Jours fériés USA + spécifiques région PJM
        usa_holidays = holidays.UnitedStates()
        
        holiday_list = []
        for year in range(start_year, end_year + 1):
            try:
                # Récupération holidays pour l'année
                year_holidays = usa_holidays[year]
                
                # Vérification type retourné
                if isinstance(year_holidays, dict):
                    # Format dictionnaire normal
                    for date, name in year_holidays.items():
                        holiday_list.append({
                            self.config.data.datetime_col: pd.Timestamp(date),
                            'holiday_name': name,
                            'is_holiday': 1,
                            'holiday_type': self._classify_holiday_type(name)
                        })
                else:
                    # Format alternatif - itération directe
                    for holiday_date in usa_holidays.get_list(str(year)):
                        holiday_name = usa_holidays.get(holiday_date, f"Holiday {holiday_date}")
                        holiday_list.append({
                            self.config.data.datetime_col: pd.Timestamp(holiday_date),
                            'holiday_name': str(holiday_name),
                            'is_holiday': 1,
                            'holiday_type': self._classify_holiday_type(str(holiday_name))
                        })
                        
            except Exception as e:
                logger.warning(f"Erreur année {year}: {e}")
                # Fallback holidays manually for year
                manual_holidays = [
                    f"{year}-01-01",  # New Year
                    f"{year}-07-04",  # Independence Day  
                    f"{year}-12-25",  # Christmas
                ]
                
                for date_str in manual_holidays:
                    try:
                        holiday_list.append({
                            self.config.data.datetime_col: pd.Timestamp(date_str),
                            'holiday_name': 'Manual Holiday',
                            'is_holiday': 1,
                            'holiday_type': 'major'
                        })
                    except:
                        continue
        
        if holiday_list:
            self.holidays_data = pd.DataFrame(holiday_list)
            logger.info(f"✅ {len(holiday_list)} jours fériés générés ({start_year}-{end_year})")
        else:
            # Fallback: pas de holidays
            logger.warning("⚠️ Aucun jour férié généré - création DataFrame vide")
            self.holidays_data = pd.DataFrame(columns=[
                self.config.data.datetime_col, 'holiday_name', 'is_holiday', 'holiday_type'
            ])
    
    def _classify_holiday_type(self, holiday_name: str) -> str:
        """Classification impact business des jours fériés"""
        major_holidays = ['Christmas', 'New Year', 'Thanksgiving', 'Independence Day']
        minor_holidays = ['Labor Day', 'Memorial Day', 'Veterans Day']
        
        if any(major in holiday_name for major in major_holidays):
            return 'major'  # Impact fort sur consommation
        elif any(minor in holiday_name for minor in minor_holidays):
            return 'minor'  # Impact modéré
        else:
            return 'regular'  # Impact faible
    
    def clean_pjm_data(self) -> None:
        """Nettoyage spécifique données PJM"""
        logger.info("🧹 Nettoyage données PJM...")
        
        # Conversion datetime
        self.consumption_data[self.config.data.datetime_col] = pd.to_datetime(self.consumption_data[self.config.data.datetime_col])
        
        # Sort chronologique
        self.consumption_data = self.consumption_data.sort_values(self.config.data.datetime_col).reset_index(drop=True)
        
        # Vérification continuité temporelle
        time_diffs = self.consumption_data[self.config.data.datetime_col].diff()
        expected_freq = pd.Timedelta(hours=1)
        irregular_gaps = (time_diffs != expected_freq).sum()
        
        if irregular_gaps > 1:  # Ignore first NaT
            logger.warning(f"⚠️  {irregular_gaps-1} gaps temporels détectés")
        
        # Handle missing consumption values
        consumption_col = self.config.data.consumption_col
        missing_count = self.consumption_data[consumption_col].isna().sum()
        
        if missing_count > 0:
            logger.warning(f"⚠️  {missing_count} valeurs manquantes - interpolation linéaire")
            self.consumption_data[consumption_col] = self.consumption_data[consumption_col].interpolate(method='linear')
        
        # Remove outliers (Z-score > 3)
        z_scores = np.abs(stats.zscore(self.consumption_data[consumption_col]))
        outliers = z_scores > self.config.data.outlier_threshold
        n_outliers = outliers.sum()
        
        if n_outliers > 0:
            logger.warning(f"⚠️  {n_outliers} outliers détectés ({n_outliers/len(self.consumption_data)*100:.2f}%)")
            
            # Replace outliers with rolling median
            rolling_median = self.consumption_data[consumption_col].rolling(window=24, center=True).median()
            self.consumption_data.loc[outliers, consumption_col] = rolling_median[outliers]
        
        # Ensure positive consumption
        negative_mask = self.consumption_data[consumption_col] <= 0
        if negative_mask.sum() > 0:
            logger.warning(f"⚠️  {negative_mask.sum()} valeurs ≤0 - correction par médiane")
            median_consumption = self.consumption_data[consumption_col].median()
            self.consumption_data.loc[negative_mask, consumption_col] = median_consumption
        
        logger.info("✅ Nettoyage PJM terminé")
    
    def create_electricity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering spécifique électricité PJM"""
        logger.info("⚡ Création features électricité PJM...")
        
        df = df.copy()
        df = df.sort_values(self.config.data.datetime_col)
        
        # Basic temporal features
        dt_col = self.config.data.datetime_col
        df['year'] = df[dt_col].dt.year
        df['month'] = df[dt_col].dt.month
        df['day'] = df[dt_col].dt.day
        df['hour'] = df[dt_col].dt.hour
        df['day_of_week'] = df[dt_col].dt.dayofweek  # 0=Monday
        df['day_of_year'] = df[dt_col].dt.dayofyear
        df['week_of_year'] = df[dt_col].dt.isocalendar().week
        df['quarter'] = df[dt_col].dt.quarter
        
        # PJM business rules features
        df['is_weekend'] = df['day_of_week'].isin(PJM_BUSINESS_RULES['weekend_days']).astype(int)
        df['is_business_day'] = df['day_of_week'].isin(PJM_BUSINESS_RULES['business_days']).astype(int)
        
        # Peak/Off-peak hours (crucial pour électricité)
        df['is_peak_hour'] = df['hour'].isin(PJM_BUSINESS_RULES['peak_hours']).astype(int)
        df['is_off_peak'] = df['hour'].isin(PJM_BUSINESS_RULES['off_peak_hours']).astype(int)
        df['is_shoulder_hour'] = df['hour'].isin(PJM_BUSINESS_RULES['shoulder_hours']).astype(int)
        
        # Seasonal patterns (climatisation/chauffage)
        df['is_peak_season'] = df['month'].isin(PJM_BUSINESS_RULES['peak_season_months']).astype(int)
        df['is_heating_season'] = df['month'].isin(PJM_BUSINESS_RULES['heating_season_months']).astype(int)
        df['is_shoulder_season'] = (~df['is_peak_season'] & ~df['is_heating_season']).astype(int)
        
        # Complex time patterns
        df['is_business_peak'] = (df['is_business_day'] & df['is_peak_hour']).astype(int)
        df['is_weekend_evening'] = (df['is_weekend'] & df['is_peak_hour']).astype(int)
        
        # Cyclical encoding (important pour SARIMA)
        # Hour cycle (24h)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week cycle
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Monthly cycle
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Yearly cycle
        df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        logger.info("✅ Features temporelles PJM créées")
        return df
    
    def merge_weather_and_holidays(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fusion météo synthétique et jours fériés"""
        logger.info("🔗 Fusion données externes...")
        
        df = df.copy()
        
        # 1. Merge weather data
        if self.weather_data is not None:
            # Merge on datetime (exact match)
            df = df.merge(
                self.weather_data, 
                on=self.config.data.datetime_col, 
                how='left'
            )
            
            # Weather-consumption interaction features
            temp_col = self.config.data.temperature_col
            if temp_col in df.columns:
                # Heating/Cooling degree days (base 18°C)
                df['heating_degree_hours'] = np.maximum(18 - df[temp_col], 0)
                df['cooling_degree_hours'] = np.maximum(df[temp_col] - 22, 0)
                
                # Temperature categories
                df['is_very_cold'] = (df[temp_col] < 0).astype(int)
                df['is_cold'] = ((df[temp_col] >= 0) & (df[temp_col] < 10)).astype(int)
                df['is_mild'] = ((df[temp_col] >= 10) & (df[temp_col] < 25)).astype(int)
                df['is_hot'] = (df[temp_col] >= 25).astype(int)
                
                # Weather-time interactions
                df['cold_morning'] = (df['is_cold'] & (df['hour'].isin([6, 7, 8]))).astype(int)
                df['hot_afternoon'] = (df['is_hot'] & (df['hour'].isin([14, 15, 16]))).astype(int)
            
            logger.info("✅ Données météo mergées avec features interaction")
        
        # 2. Merge holidays
        if self.holidays_data is not None:
            # Create date column for merge
            df['date_only'] = df[self.config.data.datetime_col].dt.date
            holidays_processed = self.holidays_data.copy()
            holidays_processed['date_only'] = holidays_processed[self.config.data.datetime_col].dt.date
            
            # Merge holidays
            df = df.merge(
                holidays_processed[['date_only', 'is_holiday', 'holiday_type']], 
                on='date_only', 
                how='left'
            )
            
            # Fill missing
            df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
            df['holiday_type'] = df['holiday_type'].fillna('none')
            
            # Holiday effects (jours avant/après)
            df = df.sort_values(self.config.data.datetime_col)
            df['is_pre_holiday'] = df['is_holiday'].shift(-24, fill_value=0)  # 24h avant
            df['is_post_holiday'] = df['is_holiday'].shift(24, fill_value=0)  # 24h après
            
            # Major holidays impact
            df['is_major_holiday'] = (df['holiday_type'] == 'major').astype(int)
            
            df = df.drop('date_only', axis=1)
            logger.info("✅ Jours fériés mergés avec effets pré/post")
        
        return df
    
    def create_lag_and_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features lag et rolling statistics pour time series"""
        logger.info("🔄 Création features lag et rolling...")
        
        df = df.copy().sort_values(self.config.data.datetime_col)
        consumption_col = self.config.data.consumption_col
        
        # Lags importants pour patterns électricité
        lag_hours = [1, 2, 3, 6, 12, 24, 48, 168]  # 1h à 1 semaine
        
        for lag in lag_hours:
            df[f'consumption_lag_{lag}h'] = df[consumption_col].shift(lag)
        
        # Rolling statistics (fenêtres business)
        windows = [24, 72, 168, 720]  # 1 jour, 3 jours, 1 semaine, 1 mois
        
        for window in windows:
            window_name = f"{window}h"
            if window <= len(df):
                df[f'consumption_rolling_mean_{window_name}'] = df[consumption_col].rolling(window).mean()
                df[f'consumption_rolling_std_{window_name}'] = df[consumption_col].rolling(window).std()
                df[f'consumption_rolling_min_{window_name}'] = df[consumption_col].rolling(window).min()
                df[f'consumption_rolling_max_{window_name}'] = df[consumption_col].rolling(window).max()
                df[f'consumption_rolling_median_{window_name}'] = df[consumption_col].rolling(window).median()
        
        # Diff features (tendances)
        df['consumption_diff_1h'] = df[consumption_col].diff(1)
        df['consumption_diff_24h'] = df[consumption_col].diff(24)
        df['consumption_diff_168h'] = df[consumption_col].diff(168)
        
        # Ratio features vs historical
        df['vs_daily_avg'] = df[consumption_col] / (df['consumption_rolling_mean_24h'] + 1e-8)
        df['vs_weekly_avg'] = df[consumption_col] / (df['consumption_rolling_mean_168h'] + 1e-8)
        
        # Peak detection
        df['is_daily_peak'] = (df[consumption_col] == df['consumption_rolling_max_24h']).astype(int)
        df['is_weekly_peak'] = (df[consumption_col] == df['consumption_rolling_max_168h']).astype(int)
        
        logger.info("✅ Features lag et rolling créées")
        return df
    
    def detect_consumption_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Détection patterns spécifiques consommation électrique"""
        logger.info("📈 Détection patterns consommation...")
        
        df = df.copy()
        consumption_col = self.config.data.consumption_col
        
        # Load profiling
        if 'consumption_rolling_mean_24h' in df.columns:
            daily_avg = df['consumption_rolling_mean_24h'].fillna(df[consumption_col].median())
            daily_std = df['consumption_rolling_std_24h'].fillna(df[consumption_col].std())
            
            # Peak threshold dynamique
            peak_threshold = daily_avg + (BUSINESS_METRICS['peak_threshold_multiplier'] * daily_std)
            df['is_consumption_peak'] = (df[consumption_col] > peak_threshold).astype(int)
            
            # Base load (minimum daily)
            df['base_load'] = df['consumption_rolling_min_24h'].fillna(df[consumption_col].min())
            
            # Load factor
            df['load_factor'] = daily_avg / (df['consumption_rolling_max_24h'] + 1e-8)
            
            # Variability index
            df['demand_variability'] = daily_std / (daily_avg + 1e-8)
            
            # Peak intensity (combien au-dessus de normal)
            df['peak_intensity'] = (df[consumption_col] - daily_avg) / (daily_std + 1e-8)
        
        # Seasonal intensity patterns
        monthly_avg = df.groupby('month')[consumption_col].transform('mean')
        df['vs_seasonal_avg'] = df[consumption_col] / (monthly_avg + 1e-8)
        
        logger.info(f"✅ Patterns détectés - {df['is_consumption_peak'].sum() if 'is_consumption_peak' in df.columns else 0} pics identifiés")
        return df
    
    def prepare_for_sarima(self) -> Dict[str, pd.DataFrame]:
        """Pipeline complet préparation SARIMA"""
        logger.info("📊 Pipeline complet preprocessing PJM...")
        
        # 1. Chargement données
        self.load_pjm_data()
        
        # 2. Nettoyage
        self.clean_pjm_data()
        
        # 3. Feature engineering complet
        processed_data = self.consumption_data.copy()
        processed_data = self.create_electricity_features(processed_data)
        processed_data = self.merge_weather_and_holidays(processed_data)
        processed_data = self.create_lag_and_rolling_features(processed_data)
        processed_data = self.detect_consumption_patterns(processed_data)
        
        # 4. Set datetime as index
        processed_data = processed_data.set_index(self.config.data.datetime_col).sort_index()
        
        # 5. Remove NaN rows (from lags and rolling)
        initial_len = len(processed_data)
        processed_data = processed_data.dropna()
        final_len = len(processed_data)
        removed = initial_len - final_len
        
        logger.info(f"⚠️ {removed} lignes supprimées (NaN from lags/rolling)")
        
        # 6. Train/Test split
        test_hours = self.config.model.test_size_days * 24
        
        if len(processed_data) > test_hours:
            train_data = processed_data.iloc[:-test_hours].copy()
            test_data = processed_data.iloc[-test_hours:].copy()
        else:
            split_idx = int(0.8 * len(processed_data))
            train_data = processed_data.iloc[:split_idx].copy()
            test_data = processed_data.iloc[split_idx:].copy()
        
        # 7. Prepare datasets for models
        sarima_data = {
            'train': train_data[self.config.data.consumption_col],  # Série univariée
            'test': test_data[self.config.data.consumption_col],
            'train_full': train_data,  # DataFrame complet avec features
            'test_full': test_data,
            'full_processed': processed_data,
            'region': self.region,
            'date_range': {
                'start': processed_data.index.min(),
                'end': processed_data.index.max(),
                'train_end': train_data.index.max(),
                'test_start': test_data.index.min()
            }
        }
        
        logger.info(f"✅ Données SARIMA préparées:")
        logger.info(f"  Région: {self.region}")
        logger.info(f"  Training: {len(train_data):,} points ({len(train_data)/24:.1f} jours)")
        logger.info(f"  Test: {len(test_data):,} points ({len(test_data)/24:.1f} jours)")
        logger.info(f"  Features: {len(processed_data.columns)} colonnes")
        logger.info(f"  Période: {sarima_data['date_range']['start']} → {sarima_data['date_range']['end']}")
        
        return sarima_data
    
    def save_processed_data(self, processed_data: Dict) -> None:
        """Sauvegarde données preprocessées"""
        logger.info("💾 Sauvegarde données preprocessées...")
        
        # Save main processed data
        processed_file = self.config.processed_data_file
        with open(processed_file, 'wb') as f:
            pickle.dump(processed_data, f)
        
        # Save metadata
        consumption_col = self.config.data.consumption_col
        train_data = processed_data['train']
        test_data = processed_data['test']
        full_data = processed_data['full_processed']
        
        metadata = {
            'preprocessing_date': datetime.now().isoformat(),
            'region': self.region,
            'dataset_info': {
                'total_hours': len(full_data),
                'total_days': len(full_data) / 24,
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'features_count': len(full_data.columns),
                'date_range': {
                    'start': full_data.index.min().isoformat(),
                    'end': full_data.index.max().isoformat()
                }
            },
            'consumption_stats': {
                'mean_mw': float(full_data[consumption_col].mean()),
                'median_mw': float(full_data[consumption_col].median()),
                'std_mw': float(full_data[consumption_col].std()),
                'min_mw': float(full_data[consumption_col].min()),
                'max_mw': float(full_data[consumption_col].max()),
                'peak_to_avg_ratio': float(full_data[consumption_col].max() / full_data[consumption_col].mean())
            },
            'patterns_detected': {
                'total_peaks': int(full_data['is_consumption_peak'].sum()) if 'is_consumption_peak' in full_data.columns else 0,
                'peak_percentage': float(full_data['is_consumption_peak'].mean() * 100) if 'is_consumption_peak' in full_data.columns else 0,
                'avg_weekend_consumption': float(full_data[full_data['is_weekend'] == 1][consumption_col].mean()),
                'avg_weekday_consumption': float(full_data[full_data['is_weekend'] == 0][consumption_col].mean())
            },
            'business_insights': self._generate_business_insights(full_data)
        }
        
        # Save metadata
        import json
        metadata_file = Path(self.config.data.processed_data_path) / f"preprocessing_metadata_{self.region}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Sauvegarde terminée:")
        logger.info(f"  Data: {processed_file}")
        logger.info(f"  Metadata: {metadata_file}")
    
    def _generate_business_insights(self, df: pd.DataFrame) -> Dict:
        """Génération insights business automatiques"""
        consumption_col = self.config.data.consumption_col
        
        # Peak analysis
        peak_hours = df.groupby('hour')[consumption_col].mean().sort_values(ascending=False)
        off_peak_hours = peak_hours.tail(6)
        
        # Seasonal analysis
        monthly_avg = df.groupby('month')[consumption_col].mean()
        peak_month = monthly_avg.idxmax()
        low_month = monthly_avg.idxmin()
        
        # Weekend vs weekday
        weekend_avg = df[df['is_weekend'] == 1][consumption_col].mean()
        weekday_avg = df[df['is_weekend'] == 0][consumption_col].mean()
        
        insights = {
            'peak_hour': int(peak_hours.index[0]),
            'off_peak_hour': int(off_peak_hours.index[0]),
            'peak_month': int(peak_month),
            'low_consumption_month': int(low_month),
            'weekend_vs_weekday_ratio': float(weekend_avg / weekday_avg),
            'seasonal_variation_pct': float((monthly_avg.max() - monthly_avg.min()) / monthly_avg.mean() * 100),
            'daily_variation_pct': float((peak_hours.iloc[0] - peak_hours.iloc[-1]) / peak_hours.mean() * 100)
        }
        
        return insights
    
    def run_full_preprocessing(self) -> Dict:
        """Pipeline preprocessing complet"""
        logger.info("🚀 DÉBUT PREPROCESSING PJM")
        logger.info("=" * 50)
        
        # Preprocessing complet
        processed_data = self.prepare_for_sarima()
        
        # Sauvegarde
        self.save_processed_data(processed_data)
        
        # Résumé final
        logger.info("\n📊 RÉSUMÉ PREPROCESSING")
        logger.info("=" * 40)
        logger.info(f"Région: {self.region}")
        logger.info(f"Données: {len(processed_data['full_processed']):,} heures")
        logger.info(f"Période: {processed_data['date_range']['start'].strftime('%Y-%m-%d')} → {processed_data['date_range']['end'].strftime('%Y-%m-%d')}")
        logger.info(f"Training: {len(processed_data['train']):,} points")
        logger.info(f"Test: {len(processed_data['test']):,} points")
        
        logger.info("\n✅ PREPROCESSING TERMINÉ")
        logger.info("🎯 Prêt pour model training!")
        
        return processed_data

def main():
    """Point d'entrée principal"""
    try:
        # Tu peux changer la région ici
        region = "PJME"  # Options: PJME, PJMW, PJM_Load, AEP, COMED
        
        preprocessor = PJMPreprocessor(region=region)
        processed_data = preprocessor.run_full_preprocessing()
        
        print(f"\n🎯 DONNÉES {region} PRÊTES POUR MODÈLES!")
        print("Lancer: python src/model_training.py")
        
        return processed_data
    
    except Exception as e:
        logger.error(f"❌ Erreur preprocessing: {e}")
        raise

if __name__ == "__main__":
    main()