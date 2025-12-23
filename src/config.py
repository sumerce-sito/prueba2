"""
Configuraciones globales del proyecto FlightOnTime

Este módulo centraliza todas las constantes y configuraciones
utilizadas en el pipeline de datos y modelado.
"""

import os
from pathlib import Path

# ==========================================
# RUTAS DEL PROYECTO
# ==========================================

# Directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Rutas de datos
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Rutas de notebooks
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Rutas de modelos
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

# Rutas de outputs
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"

# ==========================================
# CONFIGURACIÓN DEL DATASET
# ==========================================

# Nombre del archivo de datos crudo
RAW_DATA_FILENAME = "flight_data_2024.csv"

# Umbral para definir retraso (en minutos)
DELAY_THRESHOLD = 15

# Límite de registros para evitar problemas de memoria en Colab
# Si el dataset supera este valor, se aplicará sampling estratificado
MAX_ROWS_FOR_TRAINING = 500000

# ==========================================
# CONFIGURACIÓN DE COLUMNAS
# ==========================================

# Columna de retraso de salida (utilizada para crear el target)
DELAY_COLUMN = "dep_delay"

# Variable objetivo
TARGET_COLUMN = "is_delayed"

# Columnas que causan data leakage (información posterior al vuelo)
# Estas columnas se eliminarán durante el preprocesamiento
LEAKAGE_COLUMNS = [
    "dep_delay",      # Retraso real de salida (usado solo para crear target)
    "arr_delay",      # Retraso de llegada
    "actual_elapsed_time",
    "air_time",
    "taxi_in",
    "taxi_out",
    "wheels_off",
    "wheels_on",
]

# Columnas categóricas esperadas
CATEGORICAL_FEATURES = [
    "airline",
    "origin",
    "dest",
    "time_slot",
]

# Columnas numéricas esperadas
NUMERIC_FEATURES = [
    "month",
    "day_of_week",
    "hour",
    "is_weekend",
]

# ==========================================
# CONFIGURACIÓN DE FEATURE ENGINEERING
# ==========================================

# Definición de franjas horarias
TIME_SLOTS = {
    "madrugada": (0, 6),
    "mañana": (6, 12),
    "tarde": (12, 18),
    "noche": (18, 24),
}

# ==========================================
# CONFIGURACIÓN DE MODELADO
# ==========================================

# Proporción de datos para test
TEST_SIZE = 0.2

# Semilla aleatoria para reproducibilidad
RANDOM_STATE = 42

# Número de jobs para procesamiento paralelo (-1 = todos los cores)
N_JOBS = -1

# Configuración de modelos
MODELS_CONFIG = {
    "logistic_regression": {
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
    },
}

# ==========================================
# CONFIGURACIÓN DE EVALUACIÓN
# ==========================================

# Métrica principal para selección de modelo
# Opciones: 'f1', 'recall', 'precision', 'accuracy'
PRIMARY_METRIC = "recall"

# Clase positiva (1 = Retrasado)
POSITIVE_CLASS = 1

# ==========================================
# CONFIGURACIÓN DE VISUALIZACIÓN
# ==========================================

# Estilo de gráficas
PLOT_STYLE = "seaborn-v0_8-darkgrid"

# Tamaño de figura por defecto
FIGURE_SIZE = (12, 6)

# DPI para guardar figuras
FIGURE_DPI = 300

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================

def ensure_directories():
    """
    Crea todos los directorios necesarios si no existen
    """
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✓ Estructura de directorios verificada")


def get_raw_data_path():
    """
    Retorna la ruta completa al archivo de datos crudo
    """
    return RAW_DATA_DIR / RAW_DATA_FILENAME


if __name__ == "__main__":
    # Prueba de configuración
    ensure_directories()
    print(f"\n📁 Directorio del proyecto: {PROJECT_ROOT}")
    print(f"📊 Archivo de datos: {get_raw_data_path()}")
    print(f"🎯 Umbral de retraso: {DELAY_THRESHOLD} minutos")
    print(f"🎲 Random state: {RANDOM_STATE}")
