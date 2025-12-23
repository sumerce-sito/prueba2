"""
Módulo de preprocesamiento de datos

Contiene funciones para:
- Carga de datos con manejo de tipos
- Creación de variable objetivo
- Limpieza y normalización
- Eliminación de data leakage
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import warnings

from src.config import (
    DELAY_COLUMN,
    TARGET_COLUMN,
    DELAY_THRESHOLD,
    LEAKAGE_COLUMNS,
    MAX_ROWS_FOR_TRAINING,
)

warnings.filterwarnings('ignore')


def load_flight_data(
    filepath: Path,
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Carga el dataset de vuelos desde CSV con manejo optimizado de tipos.
    
    Importante para Google Colab:
    - Define explícitamente dtype para columnas numéricas
    - Maneja columnas de fecha automáticamente
    - Implementa sampling si el dataset es muy grande
    
    Args:
        filepath: Ruta al archivo CSV
        sample_size: Número máximo de filas a cargar (None = todas)
        random_state: Semilla para reproducibilidad del sampling
    
    Returns:
        DataFrame con los datos cargados
    """
    print(f"📂 Cargando datos desde: {filepath}")
    
    # Primero, leer solo las primeras filas para detectar columnas
    df_sample = pd.read_csv(filepath, nrows=1000)
    
    # Detectar columnas numéricas (posibles para conversión)
    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
    
    # Forzar que dep_delay sea numérico (crítico para crear el target)
    dtype_dict = {}
    if DELAY_COLUMN in df_sample.columns:
        dtype_dict[DELAY_COLUMN] = 'float64'
    
    # Cargar el dataset completo
    try:
        if sample_size:
            # Cargar con límite de filas
            df = pd.read_csv(
                filepath,
                nrows=sample_size,
                dtype=dtype_dict,
                low_memory=False
            )
            print(f"✓ Datos cargados con límite de {sample_size:,} registros")
        else:
            # Cargar todo el dataset
            df = pd.read_csv(
                filepath,
                dtype=dtype_dict,
                low_memory=False
            )
            print(f"✓ Datos cargados: {len(df):,} registros")
            
            # Si es muy grande, aplicar sampling estratificado
            if len(df) > MAX_ROWS_FOR_TRAINING:
                print(f"⚠️  Dataset muy grande ({len(df):,} registros)")
                print(f"   Aplicando sampling a {MAX_ROWS_FOR_TRAINING:,} registros...")
                df = df.sample(n=MAX_ROWS_FOR_TRAINING, random_state=random_state)
                print(f"✓ Sampling completado")
    
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        raise
    
    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza los nombres de columnas a formato estándar.
    
    - Convierte a minúsculas
    - Reemplaza espacios por guiones bajos
    - Elimina caracteres especiales
    
    Args:
        df: DataFrame original
    
    Returns:
        DataFrame con columnas normalizadas
    """
    df = df.copy()
    
    # Normalizar nombres
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(' ', '_')
        .str.replace('[^a-z0-9_]', '', regex=True)
    )
    
    print(f"✓ Nombres de columnas normalizados")
    return df


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la variable objetivo binaria 'is_delayed'.
    
    Regla:
    - is_delayed = 1 si dep_delay > DELAY_THRESHOLD (15 minutos)
    - is_delayed = 0 en caso contrario
    
    Args:
        df: DataFrame con columna de retraso
    
    Returns:
        DataFrame con columna TARGET_COLUMN añadida
    """
    df = df.copy()
    
    if DELAY_COLUMN not in df.columns:
        raise ValueError(f"❌ Columna '{DELAY_COLUMN}' no encontrada en el dataset")
    
    # Crear variable binaria
    df[TARGET_COLUMN] = (df[DELAY_COLUMN] > DELAY_THRESHOLD).astype(int)
    
    # Estadísticas
    n_delayed = df[TARGET_COLUMN].sum()
    n_ontime = len(df) - n_delayed
    pct_delayed = n_delayed / len(df) * 100
    
    print(f"\n✓ Variable objetivo creada: '{TARGET_COLUMN}'")
    print(f"  Regla: retraso > {DELAY_THRESHOLD} minutos")
    print(f"  Distribución:")
    print(f"    - Puntuales (0): {n_ontime:,} ({100-pct_delayed:.1f}%)")
    print(f"    - Retrasados (1): {n_delayed:,} ({pct_delayed:.1f}%)")
    
    return df


def remove_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina columnas que causan data leakage.
    
    Data leakage: información que no estaría disponible en el momento
    de hacer la predicción (valores posteriores al evento).
    
    Args:
        df: DataFrame original
    
    Returns:
        DataFrame sin columnas de leakage
    """
    df = df.copy()
    
    # Filtrar solo las columnas que realmente existen
    cols_to_remove = [col for col in LEAKAGE_COLUMNS if col in df.columns]
    
    if cols_to_remove:
        df = df.drop(columns=cols_to_remove)
        print(f"\n✓ Columnas de data leakage eliminadas: {len(cols_to_remove)}")
        for col in cols_to_remove:
            print(f"    - {col}")
    else:
        print("\n✓ No se encontraron columnas de data leakage")
    
    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
    """
    Maneja valores nulos en el dataset.
    
    Estrategia 'auto':
    - Numéricas: mantener nulos (serán imputados en el pipeline)
    - Categóricas: rellenar con 'missing'
    
    Args:
        df: DataFrame con posibles valores nulos
        strategy: Estrategia de imputación ('auto', 'drop', 'fill')
    
    Returns:
        DataFrame procesado
    """
    df = df.copy()
    
    # Reporte inicial de nulos
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    
    if len(cols_with_nulls) == 0:
        print("\n✓ No se encontraron valores nulos")
        return df
    
    print(f"\n📊 Valores nulos encontrados en {len(cols_with_nulls)} columnas:")
    for col, count in cols_with_nulls.items():
        pct = count / len(df) * 100
        print(f"    - {col}: {count:,} ({pct:.1f}%)")
    
    if strategy == 'auto':
        # Rellenar categóricas con 'missing'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna('unknown')
        
        print("\n✓ Valores nulos en categóricas rellenados con 'unknown'")
        print("  (Los nulos en numéricas se manejarán en el pipeline)")
    
    elif strategy == 'drop':
        df = df.dropna()
        print(f"\n✓ Filas con nulos eliminadas. Registros restantes: {len(df):,}")
    
    return df


def detect_and_parse_dates(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Detecta y parsea automáticamente columnas de fecha/hora.
    
    Args:
        df: DataFrame original
    
    Returns:
        Tuple (DataFrame con fechas parseadas, lista de columnas de fecha)
    """
    df = df.copy()
    date_columns = []
    
    # Buscar columnas con patrones de fecha en el nombre
    potential_date_cols = [
        col for col in df.columns
        if any(keyword in col.lower() for keyword in ['date', 'time', 'day', 'year'])
    ]
    
    for col in potential_date_cols:
        try:
            # Intentar parsear como fecha
            df[col] = pd.to_datetime(df[col], errors='coerce')
            if df[col].dtype == 'datetime64[ns]':
                date_columns.append(col)
        except:
            continue
    
    if date_columns:
        print(f"\n✓ Columnas de fecha detectadas y parseadas: {date_columns}")
    
    return df, date_columns


def preprocess_data(
    filepath: Path,
    sample_size: Optional[int] = None,
    save_processed: bool = False,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Pipeline completo de preprocesamiento.
    
    Pasos:
    1. Cargar datos con tipos optimizados
    2. Normalizar nombres de columnas
    3. Detectar y parsear fechas
    4. Crear variable objetivo
    5. Eliminar columnas de data leakage
    6. Manejar valores nulos
    
    Args:
        filepath: Ruta al archivo CSV crudo
        sample_size: Límite de filas (None = todas)
        save_processed: Si guardar el resultado procesado
        output_path: Ruta para guardar (requerido si save_processed=True)
    
    Returns:
        DataFrame preprocesado
    """
    print("=" * 60)
    print("🔧 INICIANDO PREPROCESAMIENTO DE DATOS")
    print("=" * 60)
    
    # 1. Cargar datos
    df = load_flight_data(filepath, sample_size)
    
    # 2. Normalizar nombres
    df = normalize_column_names(df)
    
    # 3. Parsear fechas
    df, date_cols = detect_and_parse_dates(df)
    
    # 4. Crear variable objetivo
    df = create_target_variable(df)
    
    # 5. Eliminar data leakage
    df = remove_leakage_columns(df)
    
    # 6. Manejar nulos
    df = handle_missing_values(df, strategy='auto')
    
    # Resumen final
    print("\n" + "=" * 60)
    print("✅ PREPROCESAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"  📊 Registros finales: {len(df):,}")
    print(f"  📋 Columnas finales: {len(df.columns)}")
    print(f"  🎯 Variable objetivo: '{TARGET_COLUMN}'")
    
    # Guardar si se solicita
    if save_processed and output_path:
        df.to_csv(output_path, index=False)
        print(f"\n💾 Datos procesados guardados en: {output_path}")
    
    return df


if __name__ == "__main__":
    from src.config import get_raw_data_path, PROCESSED_DATA_DIR
    
    # Ejemplo de uso
    raw_path = get_raw_data_path()
    output_path = PROCESSED_DATA_DIR / "flight_data_processed.csv"
    
    df = preprocess_data(raw_path, save_processed=True, output_path=output_path)
    
    print("\n📋 Primeras filas:")
    print(df.head())
