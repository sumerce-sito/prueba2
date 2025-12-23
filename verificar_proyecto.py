"""
Script de verificación del proyecto FlightOnTime

Este script verifica que la estructura del proyecto esté completa
y que todos los archivos necesarios estén en su lugar.
"""

from pathlib import Path
import sys

def verificar_estructura():
    """Verifica la estructura completa del proyecto"""
    
    print("="*60)
    print("🔍 VERIFICANDO ESTRUCTURA DEL PROYECTO")
    print("="*60)
    
    # Directorio raíz del proyecto
    project_root = Path(__file__).parent
    
    # Archivos y directorios requeridos
    estructura_requerida = {
        "Archivos de configuración": [
            "README.md",
            "requirements.txt",
            ".gitignore",
        ],
        "Código fuente (src/)": [
            "src/__init__.py",
            "src/config.py",
            "src/preprocessing.py",
            "src/features.py",
            "src/modeling.py",
            "src/evaluation.py",
        ],
        "Notebooks": [
            "notebooks/00_eda.ipynb",
            "notebooks/01_train_model.ipynb",
        ],
        "Directorios de datos": [
            "data/raw",
            "data/processed",
        ],
        "Directorios de modelos": [
            "models",
        ],
        "Directorios de salida": [
            "outputs/figures",
            "outputs/metrics",
        ],
    }
    
    # Verificar cada categoría
    total_items = 0
    items_encontrados = 0
    items_faltantes = []
    
    for categoria, items in estructura_requerida.items():
        print(f"\n📁 {categoria}:")
        for item in items:
            total_items += 1
            item_path = project_root / item
            
            # Verificar si existe (archivo o directorio)
            existe = item_path.exists()
            
            if existe:
                items_encontrados += 1
                tipo = "📄" if item_path.is_file() else "📂"
                print(f"  ✓ {tipo} {item}")
            else:
                items_faltantes.append(item)
                print(f"  ✗ ❌ {item} (NO ENCONTRADO)")
    
    # Verificar dataset
    print(f"\n📊 Dataset:")
    dataset_path = project_root / "data" / "raw" / "flight_data_2024.csv"
    if dataset_path.exists():
        size_mb = dataset_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ flight_data_2024.csv ({size_mb:.1f} MB)")
        items_encontrados += 1
    else:
        print(f"  ✗ ❌ flight_data_2024.csv (NO ENCONTRADO)")
        items_faltantes.append("data/raw/flight_data_2024.csv")
    total_items += 1
    
    # Resumen
    print("\n" + "="*60)
    print("📊 RESUMEN DE VERIFICACIÓN")
    print("="*60)
    print(f"  Total de items verificados: {total_items}")
    print(f"  Items encontrados: {items_encontrados}")
    print(f"  Items faltantes: {len(items_faltantes)}")
    
    if len(items_faltantes) == 0:
        print("\n✅ ¡PROYECTO COMPLETO! Todos los archivos están en su lugar.")
        print("   Listo para ejecutar los notebooks.")
        return True
    else:
        print("\n⚠️  PROYECTO INCOMPLETO. Faltan los siguientes items:")
        for item in items_faltantes:
            print(f"    - {item}")
        return False


def mostrar_siguientes_pasos():
    """Muestra los siguientes pasos para usar el proyecto"""
    
    print("\n" + "="*60)
    print("🚀 SIGUIENTES PASOS")
    print("="*60)
    
    print("\n1️⃣  Instalar dependencias:")
    print("    pip install -r requirements.txt")
    
    print("\n2️⃣  Ejecutar análisis exploratorio:")
    print("    jupyter notebook notebooks/00_eda.ipynb")
    
    print("\n3️⃣  Entrenar modelos:")
    print("    jupyter notebook notebooks/01_train_model.ipynb")
    
    print("\n4️⃣  Revisar resultados:")
    print("    - Gráficas: outputs/figures/")
    print("    - Métricas: outputs/metrics/")
    print("    - Modelo: models/model.joblib")
    
    print("\n" + "="*60)
    print("📚 Documentación completa en README.md")
    print("="*60 + "\n")


if __name__ == "__main__":
    proyecto_completo = verificar_estructura()
    
    if proyecto_completo:
        mostrar_siguientes_pasos()
    else:
        print("\n⚠️  Por favor, completa la estructura del proyecto antes de continuar.")
        sys.exit(1)
