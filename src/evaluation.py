"""
Módulo de evaluación

Contiene funciones para:
- Cálculo de métricas de clasificación
- Matrices de confusión
- Curvas ROC y Precision-Recall
- Reportes de evaluación
- Guardado de métricas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Any
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

from src.config import (
    METRICS_DIR,
    FIGURES_DIR,
    PRIMARY_METRIC,
    POSITIVE_CLASS,
    FIGURE_SIZE,
    FIGURE_DPI
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None
) -> Dict[str, float]:
    """
    Calcula todas las métricas de evaluación.
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones (clase)
        y_proba: Probabilidades de la clase positiva (opcional)
    
    Returns:
        Diccionario con todas las métricas
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0),
        'recall': recall_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0),
    }
    
    # Agregar AUC-ROC si se proporcionan probabilidades
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            metrics['avg_precision'] = average_precision_score(y_true, y_proba)
        except:
            pass
    
    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "Modelo") -> None:
    """
    Imprime las métricas de forma formateada.
    
    Args:
        metrics: Diccionario con métricas
        model_name: Nombre del modelo
    """
    print(f"\n{'='*60}")
    print(f"📊 MÉTRICAS DE EVALUACIÓN - {model_name}")
    print(f"{'='*60}")
    
    for metric_name, value in metrics.items():
        metric_display = metric_name.replace('_', ' ').title()
        print(f"  {metric_display:.<30} {value:.4f}")
    
    print(f"{'='*60}\n")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path = None,
    model_name: str = "Modelo"
) -> plt.Figure:
    """
    Crea y muestra la matriz de confusión.
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones
        save_path: Ruta para guardar la figura (opcional)
        model_name: Nombre del modelo
    
    Returns:
        Figura de matplotlib
    """
    # Calcular matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot con seaborn
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Puntual (0)', 'Retrasado (1)'],
        yticklabels=['Puntual (0)', 'Retrasado (1)'],
        ax=ax,
        cbar_kws={'label': 'Cantidad de vuelos'}
    )
    
    ax.set_xlabel('Predicción', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valor Real', fontsize=12, fontweight='bold')
    ax.set_title(f'Matriz de Confusión - {model_name}', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"💾 Matriz de confusión guardada en: {save_path}")
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Path = None,
    model_name: str = "Modelo"
) -> plt.Figure:
    """
    Crea la curva ROC.
    
    Args:
        y_true: Valores reales
        y_proba: Probabilidades de la clase positiva
        save_path: Ruta para guardar la figura (opcional)
        model_name: Nombre del modelo
    
    Returns:
        Figura de matplotlib
    """
    # Calcular curva ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Plot
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
    ax.set_title(f'Curva ROC - {model_name}', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"💾 Curva ROC guardada en: {save_path}")
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Path = None,
    model_name: str = "Modelo"
) -> plt.Figure:
    """
    Crea la curva Precision-Recall.
    
    Más informativa que ROC para datasets desbalanceados.
    
    Args:
        y_true: Valores reales
        y_proba: Probabilidades de la clase positiva
        save_path: Ruta para guardar la figura (opcional)
        model_name: Nombre del modelo
    
    Returns:
        Figura de matplotlib
    """
    # Calcular curva Precision-Recall
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Plot
    ax.plot(recall, precision, color='darkgreen', lw=2, 
            label=f'PR curve (AP = {avg_precision:.3f})')
    ax.axhline(y=y_true.mean(), color='navy', linestyle='--', 
               label=f'Baseline (prevalence = {y_true.mean():.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Sensibilidad)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title(f'Curva Precision-Recall - {model_name}', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"💾 Curva Precision-Recall guardada en: {save_path}")
    
    return fig


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path = None
) -> str:
    """
    Genera y guarda el reporte de clasificación.
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones
        save_path: Ruta para guardar el reporte (opcional)
    
    Returns:
        String con el reporte
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=['Puntual (0)', 'Retrasado (1)'],
        digits=4
    )
    
    # Guardar si se especifica ruta
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE CLASIFICACIÓN\n")
            f.write("="*60 + "\n\n")
            f.write(report)
        print(f"💾 Reporte guardado en: {save_path}")
    
    return report


def save_metrics_json(
    metrics: Dict[str, float],
    save_path: Path
) -> None:
    """
    Guarda las métricas en formato JSON.
    
    Args:
        metrics: Diccionario con métricas
        save_path: Ruta para guardar el archivo JSON
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Métricas guardadas en: {save_path}")


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Modelo",
    save_outputs: bool = True
) -> Dict[str, float]:
    """
    Evaluación completa de un modelo.
    
    Realiza:
    - Predicciones
    - Cálculo de métricas
    - Generación de gráficas
    - Guardado de resultados
    
    Args:
        model: Modelo entrenado (con método predict y predict_proba)
        X_test: Features de prueba
        y_test: Target de prueba
        model_name: Nombre del modelo
        save_outputs: Si guardar las salidas en disco
    
    Returns:
        Diccionario con métricas
    """
    print(f"\n{'='*60}")
    print(f"🔍 EVALUANDO MODELO: {model_name}")
    print(f"{'='*60}")
    
    # Realizar predicciones
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probabilidad de la clase positiva
    
    # Calcular métricas
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    
    # Imprimir métricas
    print_metrics(metrics, model_name)
    
    # Reporte de clasificación
    report = save_classification_report(
        y_test,
        y_pred,
        save_path=METRICS_DIR / f"{model_name.lower().replace(' ', '_')}_report.txt" if save_outputs else None
    )
    print(report)
    
    if save_outputs:
        # Guardar métricas en JSON
        save_metrics_json(
            metrics,
            METRICS_DIR / f"{model_name.lower().replace(' ', '_')}_metrics.json"
        )
        
        # Crear figuras
        plot_confusion_matrix(
            y_test,
            y_pred,
            save_path=FIGURES_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png",
            model_name=model_name
        )
        
        plot_roc_curve(
            y_test,
            y_proba,
            save_path=FIGURES_DIR / f"{model_name.lower().replace(' ', '_')}_roc_curve.png",
            model_name=model_name
        )
        
        plot_precision_recall_curve(
            y_test,
            y_proba,
            save_path=FIGURES_DIR / f"{model_name.lower().replace(' ', '_')}_pr_curve.png",
            model_name=model_name
        )
        
        plt.close('all')  # Cerrar todas las figuras
    
    return metrics


def compare_models(
    models_metrics: Dict[str, Dict[str, float]],
    primary_metric: str = PRIMARY_METRIC
) -> str:
    """
    Compara múltiples modelos y selecciona el mejor.
    
    Args:
        models_metrics: Diccionario {nombre_modelo: {metricas}}
        primary_metric: Métrica principal para selección
    
    Returns:
        Nombre del mejor modelo
    """
    print(f"\n{'='*60}")
    print(f"🏆 COMPARACIÓN DE MODELOS")
    print(f"{'='*60}")
    print(f"Métrica de selección: {primary_metric.upper()}\n")
    
    # Crear DataFrame para comparación
    df_comparison = pd.DataFrame(models_metrics).T
    df_comparison = df_comparison.round(4)
    
    print(df_comparison.to_string())
    
    # Seleccionar mejor modelo
    best_model = df_comparison[primary_metric].idxmax()
    best_value = df_comparison.loc[best_model, primary_metric]
    
    print(f"\n{'='*60}")
    print(f"✅ MEJOR MODELO: {best_model}")
    print(f"   {primary_metric.upper()}: {best_value:.4f}")
    print(f"{'='*60}\n")
    
    return best_model


if __name__ == "__main__":
    print("📦 Módulo de evaluación cargado correctamente")
    print(f"  Métrica principal: {PRIMARY_METRIC}")
    print(f"  Clase positiva: {POSITIVE_CLASS}")
