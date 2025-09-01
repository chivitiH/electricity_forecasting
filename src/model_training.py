"""
Electricity Consumption Forecasting - Model Training
Fichier: src/model_training.py
Modèles SARIMA optimisés pour données PJM réelles
"""

import os
import warnings

# Intel optimizations environment variables
os.environ['SKLEARNEX_VERBOSE'] = '1'  # Log Intel optimizations
os.environ['MKL_NUM_THREADS'] = '0'    # Use all cores
os.environ['OMP_NUM_THREADS'] = '0'    # Use all cores
os.environ['MKL_DYNAMIC'] = 'TRUE'     # Dynamic load balancing

# Enable Intel Extension for Scikit-learn
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print("✅ Intel Extension for Scikit-learn activé")
except ImportError:
    print("⚠️ Intel Extension non disponible - installation standard")

# Intel optimizations for NumPy/Pandas
try:
    import mkl
    mkl.set_num_threads(0)  # Use all available cores
    print(f"✅ Intel MKL configuré: {mkl.get_max_threads()} threads")
except ImportError:
    print("⚠️ Intel MKL non disponible")

"""
Electricity Consumption Forecasting - Model Training
Fichier: src/model_training.py
Modèles SARIMA optimisés pour données PJM réelles
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
import joblib

# Time series libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
import pmdarima as pm
from pmdarima import auto_arima

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

from config import config, BUSINESS_METRICS, PJM_BUSINESS_RULES

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class PJMForecaster:
    """
    Forecaster spécialisé pour données PJM
    - SARIMA multi-saisonnier (24h, 168h)
    - SARIMAX avec variables météo
    - Validation temporelle spécifique électricité
    - Métriques business (coûts, pics, load factor)
    """
    
    def __init__(self, region: str = "PJME"):
        self.config = config
        self.region = region
        self.config.switch_pjm_region(region)
        
        self.models = {}
        self.model_metrics = {}
        self.best_params = {}
        
        # Seasonal periods électricité
        self.seasonal_periods = {
            'daily': 24,      # Cycle quotidien (crucial)
            'weekly': 168,    # Cycle hebdomadaire (business days)
        }
        
        logger.info(f"PJMForecaster initialisé pour région {region}")
    
    def load_processed_data(self) -> Dict:
        """Chargement données preprocessées PJM"""
        logger.info(f"📊 Chargement données preprocessées {self.region}...")
        
        try:
            processed_file = self.config.processed_data_file
            with open(processed_file, 'rb') as f:
                processed_data = pickle.load(f)
            
            logger.info(f"✅ Données {self.region} chargées:")
            logger.info(f"  Train: {len(processed_data['train']):,} points")
            logger.info(f"  Test: {len(processed_data['test']):,} points")
            logger.info(f"  Features: {len(processed_data['train_full'].columns)} colonnes")
            logger.info(f"  Période train: {processed_data['date_range']['start']} → {processed_data['date_range']['train_end']}")
            
            return processed_data
            
        except FileNotFoundError:
            logger.error(f"❌ Données preprocessées {self.region} non trouvées")
            logger.info("💡 Lancez d'abord: python src/data_preprocessing.py")
            raise
    
    def check_stationarity(self, series: pd.Series, name: str = "Series") -> Dict:
        """Test stationnarité robuste pour électricité"""
        logger.info(f"📈 Test stationnarité: {name}")
        
        # Remove any NaN values
        clean_series = series.dropna()
        
        if len(clean_series) < 50:
            logger.warning(f"⚠️ Série trop courte pour test stationnarité: {len(clean_series)} points")
            return {'both_stationary': False}
        
        # Augmented Dickey-Fuller test
        try:
            adf_result = adfuller(clean_series)
            adf_pvalue = adf_result[1]
            adf_stationary = adf_pvalue < 0.05
        except Exception as e:
            logger.warning(f"⚠️ Erreur ADF test: {e}")
            adf_pvalue, adf_stationary = 1.0, False
        
        # KPSS test
        try:
            kpss_result = kpss(clean_series, regression='ct')
            kpss_pvalue = kpss_result[1]
            kpss_stationary = kpss_pvalue > 0.05
        except Exception as e:
            logger.warning(f"⚠️ Erreur KPSS test: {e}")
            kpss_pvalue, kpss_stationary = 0.0, False
        
        results = {
            'adf_pvalue': adf_pvalue,
            'adf_stationary': adf_stationary,
            'kpss_pvalue': kpss_pvalue, 
            'kpss_stationary': kpss_stationary,
            'both_stationary': adf_stationary and kpss_stationary,
            'series_length': len(clean_series)
        }
        
        logger.info(f"  ADF p-value: {adf_pvalue:.4f} ({'Stationnaire' if adf_stationary else 'Non-stationnaire'})")
        logger.info(f"  KPSS p-value: {kpss_pvalue:.4f} ({'Stationnaire' if kpss_stationary else 'Non-stationnaire'})")
        logger.info(f"  Conclusion: {'STATIONNAIRE' if results['both_stationary'] else 'NÉCESSITE DIFFÉRENCIATION'}")
        
        return results
    
    def find_optimal_sarima_params_manual(self, series: pd.Series, seasonal_period: int = 24) -> Tuple:
        """Recherche paramètres SARIMA sans pmdarima (compatible NumPy 2.x)"""
        logger.info(f"Recherche paramètres SARIMA manuelle (s={seasonal_period})...")
        
        clean_series = series.dropna()
        
        if len(clean_series) < 3 * seasonal_period:
            logger.warning(f"Pas assez de données pour s={seasonal_period}")
            return (1, 1, 1), (1, 1, 1, seasonal_period), float('inf')
        
        best_aic = float('inf')
        best_params = None
        best_seasonal = None
        
        # Grid search limité mais efficace pour électricité
        p_values = [0, 1, 2]
        d_values = [0, 1]  
        q_values = [0, 1, 2]
        P_values = [0, 1]
        D_values = [0, 1]
        Q_values = [0, 1]
        
        total_combinations = len(p_values) * len(d_values) * len(q_values) * len(P_values) * len(D_values) * len(Q_values)
        logger.info(f"Test {total_combinations} combinaisons de paramètres...")
        
        tested = 0
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    for P in P_values:
                        for D in D_values:
                            for Q in Q_values:
                                tested += 1
                                try:
                                    # Skip invalid combinations
                                    if p == 0 and d == 0 and q == 0:
                                        continue
                                    if P == 0 and D == 0 and Q == 0 and seasonal_period > 1:
                                        continue
                                    
                                    order = (p, d, q)
                                    seasonal_order = (P, D, Q, seasonal_period)
                                    
                                    model = SARIMAX(
                                        clean_series,
                                        order=order,
                                        seasonal_order=seasonal_order,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False
                                    )
                                    
                                    fitted = model.fit(disp=False, maxiter=50)
                                    aic = fitted.aic
                                    
                                    if aic < best_aic:
                                        best_aic = aic
                                        best_params = order
                                        best_seasonal = seasonal_order
                                        logger.info(f"  Nouveau meilleur: SARIMA{order}x{seasonal_order} - AIC: {aic:.2f}")
                                
                                except Exception:
                                    continue
                                
                                # Progress indication
                                if tested % 10 == 0:
                                    logger.info(f"  Progression: {tested}/{total_combinations}")
        
        if best_params is None:
            logger.warning("Aucun modèle valide - utilisation paramètres par défaut")
            best_params = (1, 1, 1)
            best_seasonal = (1, 1, 1, seasonal_period)
            best_aic = float('inf')
        
        logger.info(f"Meilleur SARIMA{best_params}x{best_seasonal} - AIC: {best_aic:.2f}")
        return best_params, best_seasonal, best_aic
    
    def train_sarima_model(self, train_data: pd.Series, model_name: str = "sarima_pjm") -> Any:
        """Entraînement SARIMA pour PJM"""
        logger.info(f"🤖 Entraînement {model_name} sur données {self.region}...")
        
        # Test stationnarité
        stationarity = self.check_stationarity(train_data, f"{self.region} consumption")
        
        # Recherche paramètres optimaux
        best_aic = float('inf')
        best_model = None
        best_params_info = {}
        
        # Test saisonnalités importantes pour électricité
        for period_name, period_value in self.seasonal_periods.items():
            logger.info(f"  Test saisonnalité {period_name} ({period_value}h)...")
            
            if len(train_data) > 5 * period_value:  # Assez de données
                try:
                    params, seasonal_params, aic = self.find_optimal_sarima_params(
                        train_data, seasonal_period=period_value
                    )
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_params_info = {
                            'order': params,
                            'seasonal_order': seasonal_params,
                            'period_name': period_name,
                            'aic': aic
                        }
                        logger.info(f"    ✅ Nouveau meilleur: AIC {aic:.2f}")
                        
                except Exception as e:
                    logger.warning(f"    ⚠️ Échec {period_name}: {e}")
                    continue
        
        # Entraînement du meilleur modèle
        if best_params_info:
            try:
                logger.info(f"🚀 Entraînement modèle final...")
                
                model = SARIMAX(
                    train_data,
                    order=best_params_info['order'],
                    seasonal_order=best_params_info['seasonal_order'],
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                fitted_model = model.fit(disp=False, maxiter=100)
                
                # Stockage
                self.models[model_name] = fitted_model
                self.best_params[model_name] = best_params_info
                
                logger.info(f"✅ {model_name} entraîné avec succès:")
                logger.info(f"  Ordre: SARIMA{best_params_info['order']}x{best_params_info['seasonal_order']}")
                logger.info(f"  Saisonnalité: {best_params_info['period_name']}")
                logger.info(f"  AIC final: {fitted_model.aic:.2f}")
                
                return fitted_model
                
            except Exception as e:
                logger.error(f"❌ Erreur entraînement final: {e}")
                return None
        else:
            logger.error(f"❌ Aucun paramètre valide trouvé")
            return None
    
    def train_sarimax_with_weather(self, train_data: pd.DataFrame, model_name: str = "sarimax_pjm") -> Any:
        """SARIMAX avec variables météo pour PJM"""
        logger.info(f"🌤️ Entraînement {model_name} avec météo...")
        
        consumption_col = self.config.data.consumption_col
        temp_col = self.config.data.temperature_col
        
        # Variables endogènes/exogènes
        endog = train_data[consumption_col]
        
        # Sélection variables météo disponibles
        exog_vars = []
        potential_vars = [temp_col, 'heating_degree_hours', 'cooling_degree_hours', 
                         'is_very_cold', 'is_hot', 'humidity']
        
        for var in potential_vars:
            if var in train_data.columns:
                exog_vars.append(var)
        
        if not exog_vars:
            logger.warning("⚠️ Aucune variable météo - fallback SARIMA")
            return self.train_sarima_model(endog, model_name.replace('sarimax', 'sarima'))
        
        exog = train_data[exog_vars]
        logger.info(f"Variables météo: {exog_vars}")
        
        # Paramètres optimaux (saisonnalité quotidienne pour météo)
        params, seasonal_params, _ = self.find_optimal_sarima_params(endog, seasonal_period=24)
        
        try:
            model = SARIMAX(
                endog,
                exog=exog,
                order=params,
                seasonal_order=seasonal_params,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            fitted_model = model.fit(disp=False, maxiter=100)
            
            # Stockage
            self.models[model_name] = fitted_model
            self.best_params[model_name] = {
                'order': params,
                'seasonal_order': seasonal_params,
                'exog_vars': exog_vars,
                'aic': fitted_model.aic
            }
            
            logger.info(f"✅ {model_name} entraîné avec {len(exog_vars)} variables météo")
            logger.info(f"  AIC: {fitted_model.aic:.2f}")
            
            return fitted_model
            
        except Exception as e:
            logger.error(f"❌ Erreur SARIMAX: {e}")
            return self.train_sarima_model(endog, model_name.replace('sarimax', 'sarima'))
    
    def forecast_with_confidence(self, model_name: str, steps: int = 24, 
                               exog_future: Optional[pd.DataFrame] = None) -> Dict:
        """Prédictions avec intervalles confiance"""
        if model_name not in self.models:
            logger.error(f"❌ Modèle {model_name} non trouvé")
            return {}
        
        model = self.models[model_name]
        
        try:
            # Génération prédictions
            if exog_future is not None:
                forecast_result = model.get_forecast(steps=steps, exog=exog_future)
            else:
                forecast_result = model.get_forecast(steps=steps)
            
            forecast = forecast_result.predicted_mean
            forecast_ci = forecast_result.conf_int()
            
            results = {
                'forecast': forecast,
                'lower_ci': forecast_ci.iloc[:, 0],
                'upper_ci': forecast_ci.iloc[:, 1],
                'model_name': model_name,
                'steps': steps,
                'forecast_dates': forecast.index
            }
            
            logger.info(f"✅ Prédictions {model_name}: {steps}h")
            logger.info(f"  Consommation prédite: {forecast.mean():.1f} MW (±{(forecast.max()-forecast.min())/2:.1f})")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur prédiction {model_name}: {e}")
            return {}
    
    def calculate_electricity_metrics(self, y_true: pd.Series, y_pred: pd.Series, model_name: str) -> Dict:
        """Métriques spécifiques électricité et business PJM"""
        
        # Métriques ML standard
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        # Métriques électricité
        mean_actual = y_true.mean()
        peak_actual = y_true.max()
        peak_predicted = y_pred.max()
        
        # Erreur pic (crucial pour grid planning)
        peak_error_pct = abs(peak_predicted - peak_actual) / peak_actual * 100
        
        # Précision directionnelle (montée/descente)
        direction_actual = np.sign(y_true.diff().dropna())
        direction_pred = np.sign(y_pred.diff().dropna())
        directional_accuracy = (direction_actual == direction_pred).mean() * 100
        
        # Load forecasting metrics
        peak_threshold = mean_actual * BUSINESS_METRICS['peak_threshold_multiplier']
        
        # Peak detection
        actual_peaks = (y_true > peak_threshold).sum()
        predicted_peaks = (y_pred > peak_threshold).sum()
        
        # Peak detection accuracy
        if actual_peaks > 0:
            peak_detection_accuracy = 1 - abs(predicted_peaks - actual_peaks) / actual_peaks
        else:
            peak_detection_accuracy = 1.0 if predicted_peaks == 0 else 0.0
        
        # Business impact (coûts)
        electricity_price = self.config.api.electricity_price_eur_mwh
        peak_multiplier = self.config.api.peak_cost_multiplier
        
        # Coûts sur/sous estimation
        over_prediction = np.maximum(y_pred - y_true, 0)
        under_prediction = np.maximum(y_true - y_pred, 0)
        
        over_cost = over_prediction.sum() * electricity_price  # Surproduction
        under_cost = under_prediction.sum() * electricity_price * peak_multiplier  # Manque (plus cher)
        total_error_cost = over_cost + under_cost
        
        # Metrics dict
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse), 
            'mae': float(mae),
            'mape': float(mape),
            'peak_error_pct': float(peak_error_pct),
            'directional_accuracy_pct': float(directional_accuracy),
            'peak_detection_accuracy': float(peak_detection_accuracy * 100),
            'electricity_metrics': {
                'mean_consumption_mw': float(mean_actual),
                'peak_actual_mw': float(peak_actual),
                'peak_predicted_mw': float(peak_predicted),
                'actual_peaks_count': int(actual_peaks),
                'predicted_peaks_count': int(predicted_peaks),
                'peak_threshold_mw': float(peak_threshold)
            },
            'business_impact': {
                'over_prediction_cost_eur': float(over_cost),
                'under_prediction_cost_eur': float(under_cost),
                'total_error_cost_eur': float(total_error_cost),
                'daily_error_cost_eur': float(total_error_cost / len(y_true) * 24),
                'cost_per_mw_error': float(total_error_cost / (mae * len(y_true)))
            }
        }
        
        # Stockage
        self.model_metrics[model_name] = metrics
        
        logger.info(f"📊 Métriques {model_name}:")
        logger.info(f"  MAPE: {mape:.2f}% | RMSE: {rmse:.1f} MW")
        logger.info(f"  Erreur pic: {peak_error_pct:.1f}% | Direction: {directional_accuracy:.1f}%")
        logger.info(f"  Coût erreur/jour: €{metrics['business_impact']['daily_error_cost_eur']:.0f}")
        
        return metrics
    
    def time_series_cross_validation(self, data: Dict, model_type: str = "sarima") -> Dict:
        """Validation croisée temporelle spécialisée électricité"""
        logger.info(f"🔄 Validation croisée {model_type} sur {self.region}...")
        
        train_full = data['train_full']
        consumption_col = self.config.data.consumption_col
        train_series = train_full[consumption_col]
        
        # Time series split (importantes pour données temporelles)
        n_splits = min(self.config.model.n_splits, 3)  # Limite pour vitesse
        test_size = 168  # 1 semaine de test par fold
        
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        
        cv_scores = {
            'mape_scores': [],
            'rmse_scores': [],
            'peak_error_scores': []
        }
        
        fold = 0
        for train_idx, val_idx in tscv.split(train_series):
            fold += 1
            logger.info(f"  Fold {fold}/{n_splits} - Test: {len(val_idx)} points")
            
            try:
                # Split données
                train_fold_df = train_full.iloc[train_idx]
                val_fold_df = train_full.iloc[val_idx]
                
                train_series_fold = train_fold_df[consumption_col]
                val_series_fold = val_fold_df[consumption_col]
                
                # Entraînement temporaire
                if model_type == "sarimax":
                    temp_model = self.train_sarimax_with_weather(train_fold_df, f"temp_cv_{fold}")
                    
                    # Prédiction avec météo
                    if f"temp_cv_{fold}" in self.best_params:
                        exog_vars = self.best_params[f"temp_cv_{fold}"].get('exog_vars', [])
                        if exog_vars:
                            exog_future = val_fold_df[exog_vars]
                            forecast_result = self.forecast_with_confidence(f"temp_cv_{fold}", len(val_series_fold), exog_future)
                        else:
                            forecast_result = self.forecast_with_confidence(f"temp_cv_{fold}", len(val_series_fold))
                    else:
                        forecast_result = self.forecast_with_confidence(f"temp_cv_{fold}", len(val_series_fold))
                else:
                    temp_model = self.train_sarima_model(train_series_fold, f"temp_cv_{fold}")
                    forecast_result = self.forecast_with_confidence(f"temp_cv_{fold}", len(val_series_fold))
                
                if forecast_result and 'forecast' in forecast_result:
                    y_pred = forecast_result['forecast']
                    
                    # Métriques fold
                    mape = mean_absolute_percentage_error(val_series_fold, y_pred) * 100
                    rmse = np.sqrt(mean_squared_error(val_series_fold, y_pred))
                    peak_error = abs(y_pred.max() - val_series_fold.max()) / val_series_fold.max() * 100
                    
                    cv_scores['mape_scores'].append(mape)
                    cv_scores['rmse_scores'].append(rmse)
                    cv_scores['peak_error_scores'].append(peak_error)
                    
                    logger.info(f"    MAPE: {mape:.2f}% | RMSE: {rmse:.1f} MW | Peak: {peak_error:.1f}%")
                
                # Nettoyage modèles temporaires
                temp_keys = [k for k in self.models.keys() if k.startswith(f"temp_cv_{fold}")]
                for key in temp_keys:
                    del self.models[key]
                    if key in self.best_params:
                        del self.best_params[key]
                    if key in self.model_metrics:
                        del self.model_metrics[key]
                        
            except Exception as e:
                logger.warning(f"⚠️ Fold {fold} échoué: {e}")
                continue
        
        # Moyennes validation croisée
        if cv_scores['mape_scores']:
            avg_metrics = {
                'cv_mape_mean': np.mean(cv_scores['mape_scores']),
                'cv_mape_std': np.std(cv_scores['mape_scores']),
                'cv_rmse_mean': np.mean(cv_scores['rmse_scores']),
                'cv_rmse_std': np.std(cv_scores['rmse_scores']),
                'cv_peak_error_mean': np.mean(cv_scores['peak_error_scores']),
                'cv_peak_error_std': np.std(cv_scores['peak_error_scores']),
                'cv_scores_detail': cv_scores
            }
            
            logger.info(f"✅ Validation croisée {self.region}:")
            logger.info(f"  MAPE: {avg_metrics['cv_mape_mean']:.2f}% ± {avg_metrics['cv_mape_std']:.2f}%")
            logger.info(f"  RMSE: {avg_metrics['cv_rmse_mean']:.1f} ± {avg_metrics['cv_rmse_std']:.1f} MW")
            
            return avg_metrics
        else:
            logger.error("❌ Validation croisée échouée")
            return {}
    
    def save_models(self) -> None:
        """Sauvegarde modèles PJM"""
        logger.info("💾 Sauvegarde modèles...")
        
        # Données complètes
        models_data = {
            'models': self.models,
            'best_params': self.best_params,
            'model_metrics': self.model_metrics,
            'region': self.region,
            'training_date': datetime.now().isoformat(),
            'config': {
                'seasonal_periods': self.seasonal_periods,
                'forecast_horizon': self.config.model.forecast_horizon_hours,
                'consumption_column': self.config.data.consumption_col
            }
        }
        
        # Sauvegarde principale
        models_file = self.config.model_file_path
        with open(models_file, 'wb') as f:
            joblib.dump(models_data, f)
        
        # Modèles individuels pour serving
        models_dir = Path(self.config.data.models_path)
        for model_name, model in self.models.items():
            individual_file = models_dir / f"{model_name}_{self.region}.pkl"
            with open(individual_file, 'wb') as f:
                joblib.dump(model, f)
        
        # Metadata
        metadata = {
            'region': self.region,
            'models_trained': list(self.models.keys()),
            'best_model': self.get_best_model_name(),
            'training_summary': self.get_training_summary(),
            'model_files': [f"{name}_{self.region}.pkl" for name in self.models.keys()]
        }
        
        metadata_file = models_dir / f"models_metadata_{self.region}.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ {len(self.models)} modèles sauvegardés:")
        logger.info(f"  Fichier principal: {models_file}")
        logger.info(f"  Métadonnées: {metadata_file}")
    
    def get_best_model_name(self) -> str:
        """Meilleur modèle basé sur MAPE"""
        if not self.model_metrics:
            return ""
        
        best_model = min(self.model_metrics.items(), key=lambda x: x[1]['mape'])
        return best_model[0]
    
    def get_training_summary(self) -> Dict:
        """Résumé entraînement pour portfolio"""
        summary = {
            'region': self.region,
            'total_models': len(self.models),
            'best_model': self.get_best_model_name(),
            'performance_summary': {}
        }
        
        for model_name, metrics in self.model_metrics.items():
            summary['performance_summary'][model_name] = {
                'mape_pct': round(metrics['mape'], 2),
                'rmse_mw': round(metrics['rmse'], 1),
                'peak_error_pct': round(metrics['peak_error_pct'], 1),
                'daily_cost_eur': round(metrics['business_impact']['daily_error_cost_eur'], 0),
                'directional_accuracy_pct': round(metrics['directional_accuracy_pct'], 1)
            }
        
        return summary
    
    def train_all_models(self, processed_data: Dict) -> Dict:
        """Entraînement pipeline complet PJM"""
        logger.info("🚀 ENTRAÎNEMENT MODÈLES PJM")
        logger.info("=" * 50)
        
        results = {}
        train_series = processed_data['train']
        test_series = processed_data['test']
        train_df = processed_data['train_full']
        test_df = processed_data['test_full']
        
        # 1. SARIMA quotidien (24h seasonality)
        logger.info("\n1️⃣ SARIMA quotidien (24h seasonality)")
        sarima_daily = self.train_sarima_model(train_series, "sarima_daily_pjm")
        
        if sarima_daily:
            forecast_daily = self.forecast_with_confidence("sarima_daily_pjm", len(test_series))
            if forecast_daily:
                metrics_daily = self.calculate_electricity_metrics(test_series, forecast_daily['forecast'], "sarima_daily_pjm")
                results['sarima_daily'] = {
                    'model': sarima_daily,
                    'forecast': forecast_daily,
                    'metrics': metrics_daily
                }
        
        # 2. SARIMA hebdomadaire (168h seasonality)
        logger.info("\n2️⃣ SARIMA hebdomadaire (168h seasonality)")
        # Use subset for speed (last 2 years)
        if len(train_series) > 17520:  # 2 years = 17520 hours
            train_subset = train_series.iloc[-17520:]
            logger.info("  Utilisation sous-ensemble (2 ans) pour vitesse")
        else:
            train_subset = train_series
        
        sarima_weekly = self.train_sarima_model(train_subset, "sarima_weekly_pjm")
        
        if sarima_weekly:
            forecast_weekly = self.forecast_with_confidence("sarima_weekly_pjm", len(test_series))
            if forecast_weekly:
                metrics_weekly = self.calculate_electricity_metrics(test_series, forecast_weekly['forecast'], "sarima_weekly_pjm")
                results['sarima_weekly'] = {
                    'model': sarima_weekly,
                    'forecast': forecast_weekly,
                    'metrics': metrics_weekly
                }
        
        # 3. SARIMAX avec météo
        logger.info("\n3️⃣ SARIMAX avec météo synthétique")
        sarimax_weather = self.train_sarimax_with_weather(train_df, "sarimax_weather_pjm")
        
        if sarimax_weather:
            # Préparer variables exogènes pour test
            model_params = self.best_params.get("sarimax_weather_pjm", {})
            exog_vars = model_params.get('exog_vars', [])
            
            if exog_vars:
                exog_test = test_df[exog_vars]
                forecast_weather = self.forecast_with_confidence("sarimax_weather_pjm", len(test_series), exog_test)
            else:
                forecast_weather = self.forecast_with_confidence("sarimax_weather_pjm", len(test_series))
            
            if forecast_weather:
                metrics_weather = self.calculate_electricity_metrics(test_series, forecast_weather['forecast'], "sarimax_weather_pjm")
                results['sarimax_weather'] = {
                    'model': sarimax_weather,
                    'forecast': forecast_weather,
                    'metrics': metrics_weather
                }
        
        # 4. Validation croisée (optionnel - prend du temps)
        logger.info("\n4️⃣ Validation croisée (échantillon)")
        cv_results = {}
        
        # CV sur échantillon pour vitesse
        if len(train_df) > 8760:  # Si plus d'1 an
            sample_size = 8760  # 1 an
            train_sample = {
                'train_full': train_df.iloc[-sample_size:],
                'train': train_series.iloc[-sample_size:]
            }
            logger.info(f"  Validation sur échantillon: {sample_size} points")
        else:
            train_sample = data
            logger.info("  Validation sur dataset complet")
        
        cv_sarima = self.time_series_cross_validation(train_sample, "sarima")
        if cv_sarima:
            cv_results['sarima'] = cv_sarima
        
        results['cross_validation'] = cv_results
        
        # 5. Sauvegarde
        self.save_models()
        
        # 6. Résumé final
        logger.info("\n📊 RÉSUMÉ ENTRAÎNEMENT")
        logger.info("=" * 40)
        
        best_model = self.get_best_model_name()
        if best_model and best_model in self.model_metrics:
            best_metrics = self.model_metrics[best_model]
            logger.info(f"🏆 Meilleur modèle: {best_model}")
            logger.info(f"  MAPE: {best_metrics['mape']:.2f}%")
            logger.info(f"  RMSE: {best_metrics['rmse']:.1f} MW")
            logger.info(f"  Erreur pic: {best_metrics['peak_error_pct']:.1f}%")
            logger.info(f"  Précision directionnelle: {best_metrics['directional_accuracy_pct']:.1f}%")
            logger.info(f"  Coût/jour: €{best_metrics['business_impact']['daily_error_cost_eur']:.0f}")
        
        logger.info(f"\n✅ {len(self.models)} modèles {self.region} entraînés")
        logger.info("🎯 Prêt pour API et Streamlit!")
        
        return results

def main():
    """Point d'entrée principal"""
    try:
        # Choix région PJM
        region = "PJME"  # Change ici: PJME, PJMW, PJM_Load, AEP, COMED
        
        # Entraînement
        forecaster = PJMForecaster(region=region)
        processed_data = forecaster.load_processed_data()
        results = forecaster.train_all_models(processed_data)
        
        print(f"\n🎯 MODÈLES {region} PRÊTS!")
        print("Prochaine étape: python src/api.py")
        
        return results
    
    except Exception as e:
        logger.error(f"❌ Erreur training: {e}")
        raise

if __name__ == "__main__":
    main()