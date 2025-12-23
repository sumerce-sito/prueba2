# ✈️ FlightOnTime - Predicción de Retrasos de Vuelos

# Dataset: Los datos son muy pesados para GitHub. Descárgalos aquí: https://drive.google.com/drive/folders/1Eosl5KbyiSLcvY5NEr9ztoYO47xY5C6M?usp=sharing

> **Proyecto de Data Science para Hackathon**  
> Clasificación binaria: predecir si un vuelo llegará puntual o retrasado

---

## 📋 Descripción del Proyecto

**FlightOnTime** es un sistema de predicción de retrasos en vuelos utilizando datos históricos de la aviación civil. El modelo clasifica cada vuelo como:

- **0 = Puntual** (retraso ≤ 15 minutos)
- **1 = Retrasado** (retraso > 15 minutos)

### 🎯 Objetivo

Ayudar a aerolíneas y pasajeros a anticipar retrasos mediante modelos de Machine Learning, optimizando la planificación operativa y mejorando la experiencia del usuario.

---

## 🗂️ Estructura del Proyecto

```
FlightOnTime/
│
├── README.md                       # Este archivo
├── requirements.txt                # Dependencias de Python
├── .gitignore                      # Archivos a ignorar en Git
│
├── data/
│   ├── raw/                        # Datos originales (flight_data_2024.csv)
│   └── processed/                  # Datos procesados
│
├── notebooks/
│   ├── 00_eda.ipynb               # Análisis Exploratorio de Datos
│   └── 01_train_model.ipynb       # Entrenamiento del Modelo
│
├── src/
│   ├── __init__.py                # Inicialización del paquete
│   ├── config.py                  # Configuraciones globales
│   ├── preprocessing.py           # Limpieza y preprocesamiento
│   ├── features.py                # Ingeniería de características
│   ├── modeling.py                # Pipeline de entrenamiento
│   └── evaluation.py              # Métricas y evaluación
│
├── models/
│   ├── model.joblib               # Modelo entrenado (Pipeline completo)
│   └── metadata.json              # Metadatos del modelo
│
└── outputs/
    ├── figures/                    # Gráficas del EDA
    └── metrics/                    # Métricas de evaluación
```

---

## 🚀 Inicio Rápido

### 1️⃣ Clonar el repositorio

```bash
git clone <tu-repositorio>
cd FlightOnTime
```

### 2️⃣ Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3️⃣ Colocar el dataset

Asegúrate de que el archivo CSV esté en:
```
data/raw/flight_data_2024.csv
```

### 4️⃣ Ejecutar los notebooks

1. **Exploración de Datos**:
   ```bash
   jupyter notebook notebooks/00_eda.ipynb
   ```

2. **Entrenamiento del Modelo**:
   ```bash
   jupyter notebook notebooks/01_train_model.ipynb
   ```

---

## 📊 Dataset

### Origen
Datos históricos de vuelos comerciales con información de:
- Aerolíneas
- Aeropuertos de origen y destino
- Fechas y horarios programados
- Retrasos reales (en minutos)

### Variable Objetivo

Se crea automáticamente la variable binaria **`is_delayed`**:
- `1` si `dep_delay > 15` minutos
- `0` en caso contrario

> **Nota**: La lógica de creación de la variable objetivo está documentada en `models/metadata.json`

---

## 🔧 Proceso de Modelado

### 1. Preprocesamiento
- Normalización de nombres de columnas
- Detección automática de tipos (fechas, categóricas, numéricas)
- Eliminación de variables con **data leakage** (info posterior al despegue)
- Manejo de valores nulos

### 2. Feature Engineering
Extracción de características temporales:
- `hour`: Hora del día
- `day_of_week`: Día de la semana (0=Lunes, 6=Domingo)
- `month`: Mes del año
- `is_weekend`: Indicador de fin de semana
- `time_slot`: Franja horaria (mañana/tarde/noche)

### 3. Modelos Evaluados
- **Logistic Regression** (baseline)
- **Random Forest Classifier** (modelo principal)

### 4. Pipeline Completo
```python
Pipeline([
    ('preprocessor', ColumnTransformer([...])),
    ('classifier', RandomForestClassifier(...))
])
```

### 5. Evaluación
**Métricas principales** (priorizadas en este orden):
1. **Recall de la clase "Retrasado"** (minimizar falsos negativos)
2. **F1-Score** (balance entre precisión y recall)
3. Curva Precision-Recall
4. Accuracy

> **Justificación**: Es más crítico identificar correctamente los vuelos retrasados (alta sensibilidad) que maximizar la precisión global.

---

## 📈 Resultados

Los resultados y métricas se guardan automáticamente en:
- `outputs/metrics/classification_report.txt`
- `outputs/metrics/confusion_matrix.json`
- `outputs/figures/` (gráficas del EDA)

### Ejemplo de predicción

```python
import joblib
import pandas as pd

# Cargar modelo
model = joblib.load('models/model.joblib')

# Crear vuelo de ejemplo
vuelo_ejemplo = pd.DataFrame({
    'airline': ['American Airlines'],
    'origin': ['JFK'],
    'dest': ['LAX'],
    'month': [6],
    'day_of_week': [1],
    'hour': [14],
    'is_weekend': [0],
    'time_slot': ['tarde']
})

# Predecir
prediccion = model.predict(vuelo_ejemplo)[0]
probabilidad = model.predict_proba(vuelo_ejemplo)[0][1]

resultado = "Retrasado" if prediccion == 1 else "Puntual"
print(f"Predicción: {resultado} (probabilidad de retraso: {probabilidad:.2%})")
```

---

## 📦 Exportación del Modelo

El pipeline completo se guarda en:
- **`models/model.joblib`**: Objeto serializado con sklearn
- **`models/metadata.json`**: Información sobre:
  - Columnas esperadas
  - Versión de scikit-learn
  - Fecha de entrenamiento
  - Métricas de evaluación
  - Regla de definición del target

---

## 🧪 Optimización para Google Colab

### Manejo de Memoria
- Se define explícitamente el `dtype` al cargar el CSV
- La variable `dep_delay` se carga como numérica
- Se implementa sampling estratificado si el dataset es > 500K registros

### Compatibilidad
- Verificación de nombres de columnas
- Conversión automática a formato esperado por el modelo

---

## 👥 Público Objetivo

Este proyecto está diseñado para **estudiantes principiantes en Data Science** que buscan:
- Aprender un flujo de trabajo completo de ML
- Entender cómo estructurar un proyecto profesional
- Ganar experiencia en un contexto de hackathon

---

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+**
- **pandas**: Manipulación de datos
- **scikit-learn**: Modelado y evaluación
- **matplotlib/seaborn**: Visualización
- **joblib**: Serialización de modelos

---

## 📝 Licencia

Este proyecto es de código abierto y está disponible para fines educativos.

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -m 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

---

## 📧 Contacto

¿Preguntas o sugerencias? Abre un issue en el repositorio.

---

**¡Buena suerte en el hackathon! ✈️🚀**

