"""
Script de entrenamiento del modelo de predicción de retrasos de vuelos

Este script:
1. Preprocesa los datos
2. Aplica feature engineering
3. Entrena un modelo Random Forest
4. Evalúa el modelo
5. Guarda el modelo entrenado

Uso:
    python train_model.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import sklearn

# Importar módulos del proyecto
from src.config import (
    get_raw_data_path,
    ensure_directories,
    MAX_ROWS_FOR_TRAINING,
    RANDOM_STATE,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    MODEL_PATH,
    METADATA_PATH,
)
from src.preprocessing import preprocess_data
from src.features import engineer_features, select_features_for_modeling
from src.modeling import (
    create_preprocessing_pipeline,
    split_train_test,
    train_random_forest,
    save_model,
    create_model_metadata,
)
from src.evaluation import evaluate_model, print_metrics


def main():
    """
    Función principal de entrenamiento
    """
    print("=" * 70)
    print("🚀 ENTRENAMIENTO DEL MODELO DE PREDICCIÓN DE RETRASOS DE VUELOS")
    print("=" * 70)
    print(f"📅 Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    start_time = time.time()
    
    # 1. Verificar estructura de directorios
    print("📁 Verificando estructura de directorios...")
    ensure_directories()
    print()
    
    # 2. Preprocesar datos (carga + limpieza + target)
    print("📊 Preprocesando datos...")
    data_path = get_raw_data_path()
    print(f"   Archivo: {data_path}")
    print(f"   Límite de registros: {MAX_ROWS_FOR_TRAINING:,}")
    print("   (Esto puede tomar algunos minutos...)\n")
    
    df = preprocess_data(data_path, sample_size=MAX_ROWS_FOR_TRAINING)
    print(f"\n   ✓ Datos procesados: {len(df):,} registros")
    print(f"   ✓ Retrasos: {df[TARGET_COLUMN].sum():,} ({df[TARGET_COLUMN].mean()*100:.1f}%)")
    print(f"   ✓ A tiempo: {(~df[TARGET_COLUMN].astype(bool)).sum():,} ({(1-df[TARGET_COLUMN].mean())*100:.1f}%)\n")

    
    # 3. Feature engineering
    print("\n⚙️  Aplicando feature engineering...\n")
    df = engineer_features(df, date_column='fl_date', create_interactions=False)
    print()
    
    # 4. Seleccionar features
    print("\n🔍 Seleccionando features para modelado...\n")
    X, y, cat_features, num_features = select_features_for_modeling(df, TARGET_COLUMN)
    print()
    
    # 5. Dividir datos
    print("✂️  Dividiendo datos en train/test...")
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    print(f"   ✓ Train: {len(X_train):,} registros")
    print(f"   ✓ Test: {len(X_test):,} registros\n")
    
    # 6. Crear pipeline de preprocesamiento
    print("🔧 Creando pipeline de preprocesamiento...")
    preprocessor = create_preprocessing_pipeline(cat_features, num_features)
    print("   ✓ Pipeline creado\n")
    
    # 7. Entrenar modelo
    print("🤖 Entrenando modelo Random Forest...")
    print("   (Esto puede tomar varios minutos...)")
    model = train_random_forest(X_train, y_train, preprocessor)
    print("   ✓ Modelo entrenado\n")
    
    # 8. Evaluar modelo
    print("📈 Evaluando modelo...")
    metrics = evaluate_model(model, X_test, y_test)
    print_metrics(metrics)
    print()
    
    # 9. Crear metadatos
    print("📝 Creando metadatos...")
    metadata = create_model_metadata(
        model_name="RandomForest",
        categorical_features=cat_features,
        numeric_features=num_features,
        metrics=metrics,
        sklearn_version=sklearn.__version__,
    )
    metadata["training_samples"] = len(X_train)
    metadata["test_samples"] = len(X_test)
    metadata["training_date"] = datetime.now().isoformat()
    print("   ✓ Metadatos creados\n")
    
    # 10. Guardar modelo
    print("💾 Guardando modelo...")
    save_model(model, MODEL_PATH, METADATA_PATH, metadata)
    print(f"   ✓ Modelo guardado: {MODEL_PATH}")
    print(f"   ✓ Metadatos guardados: {METADATA_PATH}\n")
    
    # Resumen final
    elapsed_time = time.time() - start_time
    print("=" * 70)
    print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"⏱️  Tiempo total: {elapsed_time/60:.2f} minutos")
    print(f"📊 Accuracy: {metrics['accuracy']:.3f}")
    print(f"🎯 Recall: {metrics['recall']:.3f}")
    print(f"📍 Precision: {metrics['precision']:.3f}")
    print(f"🔢 F1-Score: {metrics['f1_score']:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR DURANTE EL ENTRENAMIENTO:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
