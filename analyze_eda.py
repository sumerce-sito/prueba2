"""
Script para ejecutar EDA completo y generar conclusiones detalladas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
from pathlib import Path

warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Importar módulos del proyecto
import sys
sys.path.append('.')

from src.config import get_raw_data_path, DELAY_THRESHOLD, TARGET_COLUMN
from src.preprocessing import load_flight_data, normalize_column_names, create_target_variable

print("="*80)
print("📊 ANÁLISIS EXPLORATORIO DE DATOS - FlightOnTime")
print("="*80)

# 1. CARGAR DATOS
print("\n1️⃣ CARGANDO DATASET...")
raw_path = get_raw_data_path()

# Cargar muestra representativa para análisis rápido
df = load_flight_data(raw_path, sample_size=100000)  # 100K para análisis rápido

# 2. PREPROCESAMIENTO BÁSICO
print("\n2️⃣ PREPROCESANDO DATOS...")
df = normalize_column_names(df)
df = create_target_variable(df)

# 3. ANÁLISIS GENERAL
print("\n3️⃣ ANÁLISIS GENERAL DEL DATASET")
print(f"   • Dimensiones: {df.shape[0]:,} registros × {df.shape[1]} columnas")
print(f"   • Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print(f"   • Periodo: {df['fl_date'].min()} a {df['fl_date'].max()}" if 'fl_date' in df.columns else "")

# 4. ANÁLISIS DE VARIABLE OBJETIVO
print("\n4️⃣ DISTRIBUCIÓN DE LA VARIABLE OBJETIVO")
target_counts = df[TARGET_COLUMN].value_counts().sort_index()
target_pcts = df[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100

print(f"\n   Clase 0 (Puntual):   {target_counts[0]:,} vuelos ({target_pcts[0]:.2f}%)")
print(f"   Clase 1 (Retrasado): {target_counts[1]:,} vuelos ({target_pcts[1]:.2f}%)")
print(f"\n   ⚖️  Ratio de desbalance: {target_pcts.max() / target_pcts.min():.2f}:1")

# 5. ESTADÍSTICAS DE RETRASOS
print("\n5️⃣ ESTADÍSTICAS DE RETRASOS")
if 'dep_delay' in df.columns:
    delayed = df[df['dep_delay'] > 0]
    
    print(f"\n   Total de vuelos con retraso: {len(delayed):,} ({len(delayed)/len(df)*100:.2f}%)")
    print(f"\n   📈 Retrasos (solo vuelos con retraso > 0):")
    print(f"      • Media:        {delayed['dep_delay'].mean():.1f} minutos")
    print(f"      • Mediana:      {delayed['dep_delay'].median():.1f} minutos")
    print(f"      • Desv. Est.:   {delayed['dep_delay'].std():.1f} minutos")
    print(f"      • Percentil 75: {delayed['dep_delay'].quantile(0.75):.1f} minutos")
    print(f"      • Percentil 90: {delayed['dep_delay'].quantile(0.90):.1f} minutos")
    print(f"      • Percentil 95: {delayed['dep_delay'].quantile(0.95):.1f} minutos")
    print(f"      • Máximo:       {delayed['dep_delay'].max():.1f} minutos")

# 6. ANÁLISIS POR AEROLÍNEA
print("\n6️⃣ ANÁLISIS POR AEROLÍNEA")
airline_col = None
for col in ['airline', 'carrier', 'op_carrier', 'op_unique_carrier']:
    if col in df.columns:
        airline_col = col
        break

if airline_col:
    airline_stats = df.groupby(airline_col).agg({
        TARGET_COLUMN: ['mean', 'count']
    }).reset_index()
    airline_stats.columns = ['Aerolínea', 'Tasa_Retraso', 'Total_Vuelos']
    airline_stats['Tasa_Retraso'] = (airline_stats['Tasa_Retraso'] * 100).round(2)
    airline_stats = airline_stats[airline_stats['Total_Vuelos'] >= 100]
    airline_stats = airline_stats.sort_values('Tasa_Retraso', ascending=False)
    
    print(f"\n   Total de aerolíneas: {len(airline_stats)}")
    print(f"\n   🏆 TOP 5 AEROLÍNEAS CON MAYOR TASA DE RETRASO:")
    for i, row in airline_stats.head(5).iterrows():
        print(f"      {row['Aerolínea']:.<30} {row['Tasa_Retraso']:>6.2f}% ({row['Total_Vuelos']:>6,} vuelos)")
    
    print(f"\n   ✅ TOP 5 AEROLÍNEAS MÁS PUNTUALES:")
    for i, row in airline_stats.tail(5).iterrows():
        print(f"      {row['Aerolínea']:.<30} {row['Tasa_Retraso']:>6.2f}% ({row['Total_Vuelos']:>6,} vuelos)")
    
    # Guardar para reporte
    airline_stats_dict = {
        'peores': airline_stats.head(10).to_dict('records'),
        'mejores': airline_stats.tail(10).to_dict('records'),
        'promedio_general': float(airline_stats['Tasa_Retraso'].mean())
    }
else:
    airline_stats_dict = None
    print("   ⚠️  No se encontró columna de aerolínea")

# 7. ANÁLISIS TEMPORAL
print("\n7️⃣ ANÁLISIS TEMPORAL")

# Extraer features temporales si existen
if 'fl_date' in df.columns:
    df['fl_date'] = pd.to_datetime(df['fl_date'], errors='coerce')
    df['hour'] = df['fl_date'].dt.hour if 'hour' not in df.columns else df['hour']
    df['day_of_week'] = df['fl_date'].dt.dayofweek if 'day_of_week' not in df.columns else df['day_of_week']
    df['month'] = df['fl_date'].dt.month if 'month' not in df.columns else df['month']

# Por hora del día
if 'hour' in df.columns:
    hour_stats = df.groupby('hour')[TARGET_COLUMN].mean() * 100
    peak_hour = hour_stats.idxmax()
    lowest_hour = hour_stats.idxmin()
    
    print(f"\n   ⏰ ANÁLISIS POR HORA DEL DÍA:")
    print(f"      • Hora con MÁS retrasos:  {peak_hour:02d}:00 ({hour_stats[peak_hour]:.2f}%)")
    print(f"      • Hora con MENOS retrasos: {lowest_hour:02d}:00 ({hour_stats[lowest_hour]:.2f}%)")
    print(f"      • Promedio general:        {hour_stats.mean():.2f}%")
    
    # Franjas horarias
    madrugada = hour_stats[0:6].mean()
    mañana = hour_stats[6:12].mean()
    tarde = hour_stats[12:18].mean()
    noche = hour_stats[18:24].mean()
    
    print(f"\n   🌅 ANÁLISIS POR FRANJAS HORARIAS:")
    print(f"      • Madrugada (00-06): {madrugada:.2f}%")
    print(f"      • Mañana (06-12):    {mañana:.2f}%")
    print(f"      • Tarde (12-18):     {tarde:.2f}%")
    print(f"      • Noche (18-24):     {noche:.2f}%")
    
    hour_stats_dict = {
        'por_hora': hour_stats.to_dict(),
        'pico': int(peak_hour),
        'minimo': int(lowest_hour),
        'franjas': {
            'madrugada': float(madrugada),
            'mañana': float(mañana),
            'tarde': float(tarde),
            'noche': float(noche)
        }
    }
else:
    hour_stats_dict = None

# Por día de la semana
if 'day_of_week' in df.columns:
    dow_stats = df.groupby('day_of_week')[TARGET_COLUMN].mean() * 100
    dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    
    print(f"\n   📅 ANÁLISIS POR DÍA DE LA SEMANA:")
    for day, rate in dow_stats.items():
        emoji = "📈" if rate > dow_stats.mean() else "📉"
        print(f"      {emoji} {dias[day]:.<12} {rate:>6.2f}%")
    
    semana = dow_stats[0:5].mean()
    fin_semana = dow_stats[5:7].mean()
    print(f"\n      Días de semana:  {semana:.2f}%")
    print(f"      Fin de semana:   {fin_semana:.2f}%")
    
    dow_stats_dict = {
        'por_dia': {dias[i]: float(v) for i, v in dow_stats.items()},
        'semana_vs_finde': {
            'semana': float(semana),
            'fin_semana': float(fin_semana)
        }
    }
else:
    dow_stats_dict = None

# Por mes
if 'month' in df.columns:
    month_stats = df.groupby('month')[TARGET_COLUMN].mean() * 100
    meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
             'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    
    print(f"\n   📆 ANÁLISIS POR MES:")
    for month, rate in month_stats.items():
        emoji = "🔴" if rate > month_stats.mean() + 2 else "🟢" if rate < month_stats.mean() - 2 else "🟡"
        print(f"      {emoji} {meses[month-1]:.<12} {rate:>6.2f}%")
    
    month_stats_dict = {m: float(r) for m, r in zip([meses[i-1] for i in month_stats.index], month_stats.values)}
else:
    month_stats_dict = None

# 8. CORRELACIONES
print("\n8️⃣ CORRELACIONES CON RETRASO")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if TARGET_COLUMN in numeric_cols and len(numeric_cols) > 1:
    correlations = df[numeric_cols].corr()[TARGET_COLUMN].sort_values(ascending=False)
    correlations = correlations[correlations.index != TARGET_COLUMN]
    
    print(f"\n   Top 5 variables MÁS correlacionadas con retraso:")
    for var, corr in correlations.head(5).items():
        print(f"      • {var:.<30} {corr:>7.4f}")

# 9. GUARDAR RESULTADOS
print("\n9️⃣ GUARDANDO RESULTADOS...")

# Crear reporte JSON
reporte = {
    'dataset': {
        'registros': int(len(df)),
        'columnas': int(df.shape[1]),
        'periodo': f"{df['fl_date'].min()} a {df['fl_date'].max()}" if 'fl_date' in df.columns else "N/A"
    },
    'variable_objetivo': {
        'puntuales': int(target_counts[0]),
        'retrasados': int(target_counts[1]),
        'porcentaje_retrasados': float(target_pcts[1]),
        'ratio_desbalance': float(target_pcts.max() / target_pcts.min())
    },
    'estadisticas_retrasos': {
        'media': float(delayed['dep_delay'].mean()),
        'mediana': float(delayed['dep_delay'].median()),
        'desv_std': float(delayed['dep_delay'].std()),
        'p75': float(delayed['dep_delay'].quantile(0.75)),
        'p90': float(delayed['dep_delay'].quantile(0.90)),
        'p95': float(delayed['dep_delay'].quantile(0.95))
    } if 'dep_delay' in df.columns else None,
    'analisis_aerolineas': airline_stats_dict,
    'analisis_temporal': {
        'por_hora': hour_stats_dict,
        'por_dia_semana': dow_stats_dict,
        'por_mes': month_stats_dict
    }
}

# Guardar JSON
output_path = Path('outputs/metrics/eda_resultados.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(reporte, f, indent=2, ensure_ascii=False)

print(f"   ✅ Reporte guardado: {output_path}")

print("\n" + "="*80)
print("✅ ANÁLISIS COMPLETADO")
print("="*80)
print("\nRevisa el archivo 'conclusiones_eda.md' para el reporte completo")
