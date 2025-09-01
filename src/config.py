"""
Electricity Consumption Forecasting - Configuration Centrale
Fichier: src/config.py
Adapté pour dataset PJM réel de Kaggle + Optimisations Intel
"""

# Intel optimizations (configuration corrigée)
import os
os.environ['SKLEARNEX_VERBOSE'] = 'INFO'  # Fix warning
os.environ['MKL_NUM_THREADS'] = '16'      # Fix: spécifie 16 au lieu de 0
os.environ['OMP_NUM_THREADS'] = '16'      # Fix: spécifie 16 au lieu de 0
os.environ['MKL_DYNAMIC'] = 'TRUE'

# Intel Extensions
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print("Intel Extension for Scikit-learn activé")
except ImportError:
    print("Intel Extension non disponible")

try:
    import mkl
    # Fix: utilise get_max_threads() au lieu de set_num_threads(0)
    max_threads = mkl.get_max_threads()
    mkl.set_num_threads(max_threads)  # Utilise tous les threads disponibles
    print(f"Intel MKL configuré: {max_threads} threads")
except ImportError:
    print("Intel MKL non disponible - installation requise")

from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration modèles SARIMA/SARIMAX pour électricité"""
    # SARIMA parameters
    max_p: int = 5
    max_d: int = 2  
    max_q: int = 5
    max_P: int = 2
    max_D: int = 1
    max_Q: int = 2
    
    # Seasonality periods (heures)
    seasonal_periods: List[int] = None
    primary_seasonality: int = 24  # Pattern quotidien
    
    # Training parameters
    test_size_days: int = 30  # 1 mois pour test
    validation_method: str = "time_series_split"
    n_splits: int = 5
    
    # Forecasting
    forecast_horizon_hours: int = 24  # Prédiction 24h
    confidence_levels: List[float] = None
    
    def __post_init__(self):
        if self.seasonal_periods is None:
            self.seasonal_periods = [24, 168, 8760]  # Jour, semaine, année
        if self.confidence_levels is None:
            self.confidence_levels = [0.80, 0.95]

@dataclass  
class DataConfig:
    """Configuration données PJM"""
    # Paths
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    models_path: str = "data/models"
    
    # ✅ FICHIERS PJM RÉELS (structure Kaggle)
    consumption_file: str = "PJME_hourly.csv"  # Principal - PJM East
    weather_file: str = "weather_synthetic.csv"  # Généré automatiquement
    holidays_file: str = "us_holidays.csv"      # Généré automatiquement
    
    # ✅ COLONNES PJM RÉELLES (selon dataset Kaggle)
    datetime_col: str = "Datetime"     # Avec majuscule dans PJM
    consumption_col: str = "PJME_MW"   # Consommation PJM East en MW
    temperature_col: str = "temperature"  # Météo synthétique
    
    # Régions PJM disponibles
    available_regions: Dict[str, Dict] = None
    selected_region: str = "PJME"  # Région par défaut
    
    def __post_init__(self):
        if self.available_regions is None:
            self.available_regions = {
                'PJME': {
                    'file': 'PJME_hourly.csv', 
                    'column': 'PJME_MW',
                    'name': 'PJM East',
                    'description': 'Région Est de PJM - principal dataset'
                },
                'PJMW': {
                    'file': 'PJMW_hourly.csv', 
                    'column': 'PJMW_MW',
                    'name': 'PJM West',
                    'description': 'Région Ouest de PJM'
                },
                'PJM_Load': {
                    'file': 'PJM_Load_hourly.csv', 
                    'column': 'PJM_Load_MW',
                    'name': 'PJM Total',
                    'description': 'Données agrégées PJM complètes'
                },
                'AEP': {
                    'file': 'AEP_hourly.csv', 
                    'column': 'AEP_MW',
                    'name': 'American Electric Power',
                    'description': 'American Electric Power Company'
                },
                'COMED': {
                    'file': 'COMED_hourly.csv', 
                    'column': 'COMED_MW',
                    'name': 'Commonwealth Edison',
                    'description': 'Commonwealth Edison (Chicago)'
                }
            }
    
    def set_region(self, region_key: str) -> None:
        """Change la région PJM à analyser"""
        if region_key in self.available_regions:
            region_info = self.available_regions[region_key]
            self.consumption_file = region_info['file']
            self.consumption_col = region_info['column']
            self.selected_region = region_key
        else:
            raise ValueError(f"Région {region_key} non disponible. Options: {list(self.available_regions.keys())}")
    
    # Preprocessing
    outlier_threshold: float = 3.0
    missing_method: str = "interpolate"
    smoothing_window: int = 3

@dataclass
class APIConfig:
    """Configuration API FastAPI"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Rate limiting
    requests_per_minute: int = 100
    
    # Model serving
    model_cache_size: int = 10
    prediction_cache_ttl: int = 300  # 5 minutes
    
    # Business metrics électricité
    electricity_price_eur_mwh: float = 85.0
    peak_cost_multiplier: float = 1.5
    grid_efficiency_factor: float = 0.92

@dataclass
class StreamlitConfig:
    """Configuration interface Streamlit"""
    page_title: str = "⚡ PJM Electricity Consumption Forecasting"
    page_icon: str = "⚡"
    layout: str = "wide"
    
    # Dashboard colors
    primary_color: str = "#FF6B35"
    secondary_color: str = "#004E89" 
    success_color: str = "#2E8B57"
    warning_color: str = "#FFD700"
    danger_color: str = "#DC143C"
    
    # Chart settings
    chart_height: int = 400
    update_interval_seconds: int = 30

class Config:
    """Configuration principale PJM Forecasting"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.api = APIConfig()
        self.streamlit = StreamlitConfig()
        
        # Environment
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = self.environment == "development"
        
        # Paths setup
        self.base_path = Path(__file__).parent.parent
        self.ensure_directories()
    
    def ensure_directories(self):
        """Création répertoires nécessaires"""
        directories = [
            self.data.raw_data_path,
            self.data.processed_data_path, 
            self.data.models_path,
            "assets",
            "logs"
        ]
        
        for directory in directories:
            Path(self.base_path / directory).mkdir(parents=True, exist_ok=True)
    
    def switch_pjm_region(self, region: str) -> None:
        """Change la région PJM à analyser"""
        self.data.set_region(region)
    
    @property
    def model_file_path(self) -> Path:
        """Chemin fichier modèles sauvegardés"""
        return self.base_path / self.data.models_path / "sarima_models.pkl"
    
    @property  
    def processed_data_file(self) -> Path:
        """Chemin données preprocessées"""
        region = self.data.selected_region
        return self.base_path / self.data.processed_data_path / f"consumption_engineered_{region}.pkl"

# Instance globale
config = Config()

# Business constants pour électricité
BUSINESS_METRICS = {
    "average_mw_residential": 150,
    "average_mw_industrial": 2500,
    "peak_threshold_multiplier": 1.3,  # 30% au-dessus moyenne = pic
    "co2_emission_kg_per_mwh": 350,
    "grid_loss_percentage": 8,
    "renewable_percentage_target": 35,
    "demand_response_capacity_mw": 1000
}

# PJM-specific business rules
PJM_BUSINESS_RULES = {
    "peak_hours": [17, 18, 19, 20, 21],  # 17h-21h
    "off_peak_hours": [23, 0, 1, 2, 3, 4, 5, 6],  # 23h-6h
    "shoulder_hours": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 22],
    "peak_season_months": [6, 7, 8, 9],  # Juin-Septembre (AC)
    "heating_season_months": [12, 1, 2, 3],  # Décembre-Mars
    "business_days": [0, 1, 2, 3, 4],  # Lundi-Vendredi
    "weekend_days": [5, 6]  # Samedi-Dimanche
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "INFO", 
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "logs/electricity_forecasting.log",
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": False
        }
    }
}