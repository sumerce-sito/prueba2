"""
Módulo de modelado

Contiene funciones para:
- Creación de pipelines de transformación
- Entrenamiento de modelos
- Validación cruzada
- Guardado de modelos
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, Tuple, Any
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    TEST_SIZE,
    RANDOM_STATE,
    MODELS_CONFIG,
    MODEL_PATH,
    METADATA_PATH,
    TARGET_COLUMN,
)


def create_preprocessing_pipeline(
    categorical_features: list,
    numeric_features: list
) -> ColumnTransformer:
    """
    Crea un pipeline de preprocesamiento con ColumnTransformer.
    
    Transformaciones:
    - Categóricas: Imputación ('unknown') + OneHotEncoding
    - Numéricas: Imputación (mediana)
    
    Args:
        categorical_features: Lista de nombres de columnas categóricas
        numeric_features: Lista de nombres de columnas numéricas
    
    Returns:
        ColumnTransformer configurado
    """
    # Pipeline para variables categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Pipeline para variables numéricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    # Combinar transformadores
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numeric_transformer, numeric_features)
        ],
        remainder='drop'  # Eliminar columnas no especificadas
    )
    
    print(f"✓ Pipeline de preprocesamiento creado:")
    print(f"    - Categóricas ({len(categorical_features)}): {categorical_features}")
    print(f"    - Numéricas ({len(numeric_features)}): {numeric_features}")
    
    return preprocessor


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    
    Args:
        X: Features
        y: Target
        test_size: Proporción de datos para test
        random_state: Semilla aleatoria
        stratify: Si hacer split estratificado (recomendado para desbalance)
    
    Returns:
        Tuple (X_train, X_test, y_train, y_test)
    """
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    print(f"\n✓ Datos divididos:")
    print(f"    - Train: {len(X_train):,} registros ({(1-test_size)*100:.0f}%)")
    print(f"    - Test: {len(X_test):,} registros ({test_size*100:.0f}%)")
    
    if stratify:
        print(f"\n  Distribución en Train:")
        print(f"    {y_train.value_counts(normalize=True).mul(100).round(1).to_dict()}")
        print(f"  Distribución en Test:")
        print(f"    {y_test.value_counts(normalize=True).mul(100).round(1).to_dict()}")
    
    return X_train, X_test, y_train, y_test


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer
) -> Pipeline:
    """
    Entrena un modelo de Regresión Logística (baseline).
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        preprocessor: Pipeline de preprocesamiento
    
    Returns:
        Pipeline completo entrenado
    """
    print("\n" + "=" * 60)
    print("🔧 ENTRENANDO LOGISTIC REGRESSION (BASELINE)")
    print("=" * 60)
    
    # Crear pipeline completo
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(**MODELS_CONFIG['logistic_regression']))
    ])
    
    # Entrenar
    model.fit(X_train, y_train)
    
    print("✓ Modelo entrenado correctamente")
    
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer
) -> Pipeline:
    """
    Entrena un modelo de Random Forest.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        preprocessor: Pipeline de preprocesamiento
    
    Returns:
        Pipeline completo entrenado
    """
    print("\n" + "=" * 60)
    print("🌲 ENTRENANDO RANDOM FOREST")
    print("=" * 60)
    
    # Crear pipeline completo
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(**MODELS_CONFIG['random_forest']))
    ])
    
    # Entrenar
    model.fit(X_train, y_train)
    
    print("✓ Modelo entrenado correctamente")
    
    return model


def save_model(
    model: Pipeline,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
    metadata: Dict[str, Any] = None
) -> None:
    """
    Guarda el modelo y sus metadatos.
    
    Args:
        model: Pipeline entrenado
        model_path: Ruta para guardar el modelo
        metadata_path: Ruta para guardar los metadatos
        metadata: Diccionario con metadatos del modelo
    """
    # Crear directorios si no existen
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Guardar modelo
    joblib.dump(model, model_path)
    print(f"\n💾 Modelo guardado en: {model_path}")
    
    # Guardar metadatos si se proporcionan
    if metadata:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"📄 Metadatos guardados en: {metadata_path}")


def load_model(model_path: Path = MODEL_PATH) -> Pipeline:
    """
    Carga un modelo previamente guardado.
    
    Args:
        model_path: Ruta al archivo del modelo
    
    Returns:
        Pipeline cargado
    """
    if not model_path.exists():
        raise FileNotFoundError(f"❌ Modelo no encontrado en: {model_path}")
    
    model = joblib.load(model_path)
    print(f"✓ Modelo cargado desde: {model_path}")
    
    return model


def create_model_metadata(
    model_name: str,
    categorical_features: list,
    numeric_features: list,
    metrics: Dict[str, float],
    sklearn_version: str = None
) -> Dict[str, Any]:
    """
    Crea un diccionario de metadatos del modelo.
    
    Args:
        model_name: Nombre del modelo
        categorical_features: Lista de features categóricas
        numeric_features: Lista de features numéricas
        metrics: Diccionario con métricas de evaluación
        sklearn_version: Versión de scikit-learn
    
    Returns:
        Diccionario con metadatos
    """
    import sklearn
    
    metadata = {
        "model_name": model_name,
        "training_date": datetime.now().isoformat(),
        "sklearn_version": sklearn_version or sklearn.__version__,
        "target_variable": TARGET_COLUMN,
        "delay_threshold_minutes": 15,
        "target_rule": "is_delayed = 1 if dep_delay > 15 minutes, else 0",
        "features": {
            "categorical": categorical_features,
            "numeric": numeric_features,
            "total": len(categorical_features) + len(numeric_features)
        },
        "metrics": metrics,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE
    }
    
    return metadata


if __name__ == "__main__":
    # Ejemplo de uso
    print("📦 Módulo de modelado cargado correctamente")
    print(f"  Configuraciones disponibles: {list(MODELS_CONFIG.keys())}")
