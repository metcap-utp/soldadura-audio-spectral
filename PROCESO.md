# Proceso de Clasificación de Sonidos de Soldadura SMAW

## Resumen

Este proyecto clasifica señales de audio de soldadura SMAW (Shielded Metal Arc Welding) para identificar tres atributos del proceso:

- **Espesor de placa** (plate thickness): 3 clases
- **Tipo de electrodo** (electrode type): 4 clases  
- **Tipo de corriente** (current type): 2 clases (AC/DC)

Se utiliza aprendizaje multi-tarea con características MFCC y arquitecturas de deep learning especializadas en procesamiento de audio.

---

## Pipeline General

```
Audio WAV (16 kHz)
       ↓
   Segmentación (1-50s, overlap configurable)
       ↓
   Extracción MFCC (40 coeficientes)
       ↓
   Modelo (X-Vector / ECAPA-TDNN / FeedForward)
       ↓
   Clasificación multi-tarea (3 salidas)
```

---

## Extracción de Características: MFCC

### ¿Qué son los MFCC?

Los **Mel-Frequency Cepstral Coefficients (MFCC)** son características ampliamente utilizadas en procesamiento de audio y reconocimiento de voz. Representan el espectro de potencia de una señal de audio de forma compacta, modelando la percepción auditiva humana.

### Fundamento Teórico

#### 1. Escala Mel

La escala Mel es una escala perceptual de tonos diseñada para que distancias iguales en la escala correspondan a distancias perceptuales iguales. Se define como:

```
m = 2595 × log₁₀(1 + f/700)
```

Donde:
- `m` es la frecuencia en Mels
- `f` es la frecuencia en Hz

La percepción humana del sonido es aproximadamente lineal hasta ~1 kHz y logarítmica después, lo cual la escala Mel captura.

#### 2. Banco de Filtros Mel

Se aplica un banco de filtros triangulares espaciados según la escala Mel sobre el espectro de potencia. Los filtros son más anchos en frecuencias altas y más estrechos en frecuencias bajas.

#### 3. Cepstrum

El cepstrum es la transformada de Fourier inversa del logaritmo del espectro. Permite separar componentes de excitación (pitch) de la respuesta de filtro del tracto vocal (formantes).

### Proceso de Extracción

```
Señal de audio (16 kHz)
        ↓
   Framing (25ms con hop de 10ms)
        ↓
   Ventana de Hamming
        ↓
   FFT (Fast Fourier Transform)
        ↓
   Espectro de potencia
        ↓
   Banco de filtros Mel (40 filtros)
        ↓
   Logaritmo de energía
        ↓
   DCT (Discrete Cosine Transform)
        ↓
   MFCC (40 coeficientes)
```

### Implementación en el Proyecto

```python
import librosa

mfcc = librosa.feature.mfcc(
    y=audio_segment,
    sr=16000,
    n_mfcc=40
)
```

Parámetros utilizados:
- **Sample rate**: 16000 Hz
- **n_mfcc**: 40 coeficientes
- **Frame size**: ~25ms (default librosa)
- **Hop length**: ~10ms (default librosa)

### Output

Para un segmento de audio de duración T segundos:
- Dimensiones: `(40, T×100)` aproximadamente
- 40 coeficientes MFCC
- ~100 frames por segundo

---

## Arquitecturas de Modelos

### 1. X-Vector (TDNN)

Arquitectura de referencia para speaker recognition y verificación de voz.

**Estructura:**

```
MFCC (40, time)
      ↓
TDNN Frame-level layers:
  - Conv1d(40→512, k=5, d=1)  + BN + ReLU
  - Conv1d(512→512, k=3, d=2) + BN + ReLU
  - Conv1d(512→512, k=3, d=3) + BN + ReLU
  - Conv1d(512→512, k=1, d=1) + BN + ReLU
  - Conv1d(512→1500, k=1, d=1) + BN + ReLU
      ↓
Statistics Pooling (mean + std)
      ↓
Segment-level:
  - Linear(3000→512) + BN + ReLU
      ↓
Multi-task heads:
  - Plate: Linear(512→3)
  - Electrode: Linear(512→4)
  - Current: Linear(512→2)
```

**Características clave:**
- Convoluciones dilatadas para capturar contexto temporal largo
- Statistics pooling agrega información temporal
- Embedding de 512 dimensiones

**Referencia:** Snyder et al. (2018) - "X-Vectors: Robust DNN Embeddings for Speaker Recognition"

### 2. ECAPA-TDNN

Mejora sobre X-Vector con atención y conexiones residuales.

**Componentes principales:**

1. **Res2Net Blocks**: Conexiones residuales multi-escala
2. **SE-Module (Squeeze-and-Excitation)**: Atención de canales
3. **Attentive Statistics Pooling**: Pooling con atención temporal

```
MFCC (40, time)
      ↓
TDNN Layer (40→512)
      ↓
Res2Net Blocks (×4) con SE-Module
      ↓
Multi-layer Feature Aggregation (MFA)
      ↓
Attentive Statistics Pooling
      ↓
Linear(3072→192) + BN
      ↓
Embedding (192 dim)
      ↓
Multi-task heads
```

**Ventajas sobre X-Vector:**
- Mejor captura de patrones multi-escala
- Atención adaptativa en pooling
- Mejor rendimiento con menos parámetros

**Referencia:** Desplanques et al. (2020) - "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification"

### 3. FeedForward (Baseline)

Clasificador simple para features agregados.

```
Features agregados (240 dim)
      ↓
Linear(240→512) + BN + ReLU + Dropout
      ↓
Linear(512→256) + BN + ReLU + Dropout
      ↓
Linear(256→128) + BN + ReLU + Dropout
      ↓
Multi-task heads
```

**Features agregados:**
- MFCC mean + std
- Delta MFCC mean + std
- Delta-delta MFCC mean + std
- Total: 40 × 3 × 2 = 240 dimensiones

---

## Aprendizaje Multi-Tarea (Multi-Task Learning)

### Concepto

El aprendizaje multi-tarea entrena un modelo para predecir múltiples objetivos simultáneamente, compartiendo representaciones entre tareas.

### Ventajas

1. **Eficiencia de datos**: Una señal de audio sirve para entrenar 3 tareas
2. **Regularización implícita**: Compartir representaciones reduce overfitting
3. **Transferencia de conocimiento**: Patrones útiles para una tarea benefician a las otras

### Implementación

```python
loss = (criterion(output['plate'], y_plate) + 
        criterion(output['electrode'], y_electrode) + 
        criterion(output['current'], y_current)) / 3
```

Se promedian las pérdidas de las tres tareas con igual peso.

---

## Segmentación de Audio

### Estrategia

Los archivos de audio largos se dividen en segmentos fijos:

```python
hop_size = duration × (1 - overlap)
num_segments = (audio_length - duration) / hop_size + 1
```

### Overlap

El overlap permite:
- Aumentar datos de entrenamiento
- Capturar eventos que podrían cortarse en segmentos adyacentes

**Ejemplo (duración=10s, overlap=0.5):**
```
Audio: |----seg0----|----seg1----|----seg2----|
              |----seg0----|
                    |----seg1----|
                          |----seg2----|
```

### Splits por Sesión

Para evitar data leakage, todos los segmentos de una misma sesión van al mismo conjunto (train/test/blind). Esto garantiza evaluación realista.

---

## Métricas de Evaluación

### Por Tarea

- **Accuracy**: Proporción de predicciones correctas
- **F1-Score (macro)**: Promedio de F1 por clase (robusto a desbalance)

### Global

- **Exact Match**: Porcentaje de muestras donde las 3 tareas son correctas
- **Hamming Accuracy**: Promedio de tareas correctas por muestra

### Validación Cruzada

Se utiliza **Stratified Group K-Fold** para:
- Mantener distribución de clases por fold
- Agrupar por sesión (evitar leakage)

---

## Hiperparámetros

| Parámetro | Valor |
|-----------|-------|
| Batch size | 32 |
| Epochs | 100 (early stopping: 15) |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Label smoothing | 0.1 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingWarmRestarts (T₀=10, T_mult=2) |
| SWA start | Epoch 5 |
| SWA learning rate | 1e-4 |

---

## Referencias Científicas

### MFCC y Procesamiento de Audio

1. **Davis, S., & Mermelstein, P. (1980)**. "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences." *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 28(4), 357-366.

2. **Logan, B. (2000)**. "Mel frequency cepstral coefficients for music modeling." *ISMIR*.

### X-Vector

3. **Snyder, D., Garcia-Romero, D., Sell, G., Povey, D., & Khudanpur, S. (2018)**. "X-vectors: Robust DNN embeddings for speaker recognition." *ICASSP*.

4. **Snyder, D., Garcia-Romero, D., Sell, G., McCree, A., Povey, D., & Khudanpur, S. (2019)**. "Speaker recognition for multi-speaker conversations using x-vectors." *ICASSP*.

### ECAPA-TDNN

5. **Desplanques, B., Thienpondt, J., & Demuynck, K. (2020)**. "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification." *Interspeech*.

### Multi-Task Learning

6. **Caruana, R. (1997)**. "Multitask Learning." *Machine Learning*, 28(1), 41-75.

7. **Zhang, Y., & Yang, Q. (2021)**. "A survey on multi-task learning." *IEEE Transactions on Knowledge and Data Engineering*.

### Audio en Soldadura

8. **Wang, J., et al. (2019)**. "Welding quality monitoring based on acoustic signal." *Journal of Manufacturing Processes*.

9. **Sàpiras, A., et al. (2021)**. "Deep learning for welding sound classification." *Procedia CIRP*.

---

## Recursos Adicionales

### Libros

- **Gold, B., Morgan, N., & Ellis, D. (2011)**. *Speech and Audio Signal Processing*. Wiley.
- **Rabiner, L., & Schafer, R. (2010)**. *Theory and Applications of Digital Speech Processing*. Pearson.

### Librerías

- **librosa**: https://librosa.org/ - Análisis de audio en Python
- **speechbrain**: https://speechbrain.github.io/ - Toolkit para procesamiento de voz
- **pytorch**: https://pytorch.org/ - Deep learning

### Tutoriales

- MFCC explicación visual: https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
- X-Vector tutorial: https://speechbrain.readthedocs.io/en/latest/API/speechbrain.lobes.models.Xvector.html
