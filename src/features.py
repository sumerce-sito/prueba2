"""
Módulo de ingeniería de características (Feature Engineering)

Contiene funciones para:
- Extracción de características temporales
- Creación de variables derivadas
- Transformaciones de datos
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from src.config import TIME_SLOTS


def extract_temporal_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Extrae características temporales de una columna de fecha/hora.
    
    Features extraídas:
    - hour: Hora del día (0-23)
    - day_of_week: Día de la semana (0=Lunes, 6=Domingo)
    - month: Mes del año (1-12)
    - is_weekend: 1 si es sábado o domingo, 0 en caso contrario
    
    Args:
        df: DataFrame con columna de fecha
        date_column: Nombre de la columna de fecha/hora
    
    Returns:
        DataFrame con nuevas características temporales
    """
    df = df.copy()
    
    if date_column not in df.columns:
        raise ValueError(f"❌ Columna '{date_column}' no encontrada")
    
    # Asegurar que sea tipo datetime
    if df[date_column].dtype != 'datetime64[ns]':
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Extraer componentes temporales
    df['hour'] = df[date_column].dt.hour
    df['day_of_week'] = df[date_column].dt.dayofweek  # 0=Lunes, 6=Domingo
    df['month'] = df[date_column].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    print(f"✓ Features temporales extraídas de '{date_column}':")
    print(f"    - hour (0-23)")
    print(f"    - day_of_week (0=Lunes, 6=Domingo)")
    print(f"    - month (1-12)")
    print(f"    - is_weekend (0/1)")
    
    return df


def create_time_slot(hour: int) -> str:
    """
    Asigna una franja horaria basada en la hora del día.
    
    Franjas:
    - madrugada: 0-6
    - mañana: 6-12
    - tarde: 12-18
    - noche: 18-24
    
    Args:
        hour: Hora del día (0-23)
    
    Returns:
        Nombre de la franja horaria
    """
    for slot_name, (start, end) in TIME_SLOTS.items():
        if start <= hour < end:
            return slot_name
    return 'unknown'


def add_time_slots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columna de franja horaria al DataFrame.
    
    Args:
        df: DataFrame con columna 'hour'
    
    Returns:
        DataFrame con columna 'time_slot'
    """
    df = df.copy()
    
    if 'hour' not in df.columns:
        raise ValueError("❌ Columna 'hour' no encontrada. Ejecutar extract_temporal_features primero.")
    
    df['time_slot'] = df['hour'].apply(create_time_slot)
    
    # Estadísticas de distribución
    print("\n✓ Franja horaria creada:")
    print(df['time_slot'].value_counts().sort_index())
    
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea características de interacción entre variables.
    
    Ejemplos:
    - airline_hour: Combinación de aerolínea y hora
    - route: Combinación origen-destino
    
    Args:
        df: DataFrame con variables base
    
    Returns:
        DataFrame con features de interacción (opcional, puede expandirse)
    """
    df = df.copy()
    
    # Route: combinación de origen y destino
    if 'origin' in df.columns and 'dest' in df.columns:
        df['route'] = df['origin'] + '_' + df['dest']
        print(f"✓ Feature 'route' creada: {df['route'].nunique()} rutas únicas")
    
    return df


def select_features_for_modeling(
    df: pd.DataFrame,
    target_column: str
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Selecciona y separa características para el modelado.
    
    Args:
        df: DataFrame completo
        target_column: Nombre de la columna objetivo
    
    Returns:
        Tuple (X, y, categorical_features, numeric_features)
    """
    # Separar features y target
    if target_column not in df.columns:
        raise ValueError(f"❌ Columna objetivo '{target_column}' no encontrada")
    
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # Detectar tipos de variables
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remover columnas de fecha/tiempo si aún existen
    datetime_cols = X.select_dtypes(include=['datetime64']).columns.tolist()
    if datetime_cols:
        X = X.drop(columns=datetime_cols)
        print(f"⚠️  Columnas de fecha eliminadas de X: {datetime_cols}")
    
    # Remover columnas con demasiados valores únicos en categóricas (posible ID)
    high_cardinality = []
    for col in categorical_features:
        if X[col].nunique() > 100:  # Umbral arbitrario
            high_cardinality.append(col)
    
    if high_cardinality:
        print(f"\n⚠️  Columnas categóricas con alta cardinalidad (>{100} valores únicos):")
        for col in high_cardinality:
            print(f"    - {col}: {X[col].nunique()} valores")
        print(f"  Considera eliminarlas o agruparlas")
    
    print(f"\n✓ Features seleccionadas:")
    print(f"    - Categóricas: {len(categorical_features)} columnas")
    print(f"    - Numéricas: {len(numeric_features)} columnas")
    print(f"    - Total features: {len(X.columns)}")
    print(f"    - Registros: {len(X):,}")
    
    return X, y, categorical_features, numeric_features


def engineer_features(
    df: pd.DataFrame,
    date_column: str = 'fl_date',
    create_interactions: bool = False
) -> pd.DataFrame:
    """
    Pipeline completo de feature engineering.
    
    Pasos:
    1. Extraer características temporales
    2. Crear franjas horarias
    3. (Opcional) Crear interacciones
    
    Args:
        df: DataFrame preprocesado
        date_column: Nombre de la columna de fecha
        create_interactions: Si crear features de interacción
    
    Returns:
        DataFrame con todas las características
    """
    print("=" * 60)
    print("🔧 INICIANDO FEATURE ENGINEERING")
    print("=" * 60)
    
    df = df.copy()
    
    # 1. Características temporales
    if date_column in df.columns:
        df = extract_temporal_features(df, date_column)
    else:
        print(f"⚠️  Columna '{date_column}' no encontrada. Saltando extracción temporal.")
    
    # 2. Franjas horarias
    if 'hour' in df.columns:
        df = add_time_slots(df)
    
    # 3. Interacciones (opcional)
    if create_interactions:
        df = create_interaction_features(df)
    
    print("\n" + "=" * 60)
    print("✅ FEATURE ENGINEERING COMPLETADO")
    print("=" * 60)
    
    return df


if __name__ == "__main__":
    # Ejemplo de uso
    from src.config import PROCESSED_DATA_DIR, TARGET_COLUMN
    from src.preprocessing import preprocess_data, get_raw_data_path
    
    # Cargar y preprocesar datos
    df = preprocess_data(get_raw_data_path())
    
    # Aplicar feature engineering
    df = engineer_features(df, date_column='fl_date')
    
    # Seleccionar features para modelado
    X, y, cat_features, num_features = select_features_for_modeling(df, TARGET_COLUMN)
    
    print("\n📋 Resumen de features:")
    print(f"  Categóricas: {cat_features}")
    print(f"  Numéricas: {num_features}")
