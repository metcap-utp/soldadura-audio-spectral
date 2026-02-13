# Alternativas de Extracción de Features de Audio

Este documento describe las diferentes técnicas de extracción de features basadas en análisis espectral para clasificación de audio.

## 1. MFCC (Mel-Frequency Cepstral Coefficients)

**Estado**: Baseline implementado actualmente

### Descripción
Los MFCC son coeficientes que representan el espectro de mel del audio. Son la técnica más utilizada en reconocimiento de voz y speaker recognition.

### Proceso
1. Pre-énfasis (filtro: 1 - 0.97*z^-1)
2. Framing (ventanas de 25ms con 10ms de overlap)
3. Aplicar ventana (Hamming)
4. FFT para obtener el espectro de potencia
5. Aplicar banco de filtros mel
6. Transformada discreta del coseno (DCT)
7. Liftering (compresión del rango dinámico)

### Ventajas
- Alta precisión en entornos limpios
- Buena discriminación entre clases
- Baja correlación entre coeficientes
- Computacionalmente eficientes
- Ampliamente validados en literatura

### Desventajas
- Baja robustez en ambientes ruidosos
- Pérdida de información temporal
- Alta dimensionalidad (requiere reducción)

### Referencias
- Davis & Mermelstein (1980)
- Comparative Study of Feature Extraction Techniques for Speech Recognition (Kurzekar et al., 2014)

---

## 2. Spectrogramas (STFT)

### Descripción
Representación tiempo-frecuencia del audio mediante Short-Time Fourier Transform. Se puede usar como input para CNNs 2D.

### Tipos
- **Linear spectrogram**: FFT normal
- **Mel spectrogram**: FFT con escala mel (similar a MFCC pero sin DCT)
- **Log-mel spectrogram**: Mel spectrogram con log

### Ventajas
- Representación tiempo-frecuencia completa
- Ideal para CNNs 2D
- Captura información temporal y espectral
- Estado del arte en audio classification

### Desventajas
- Alta dimensionalidad
- Requiere más datos para entrenar
- Mayor costo computacional

### Referencias
- Comprehensive Audio Analysis and Comparison of MFCCs, Spectrograms, CNNs (Cal State University)
- Audio classification with CNNs

---

## 3. GTCC (Gammatone Filter Bank Cepstral Coefficients)

### Descripción
Similar a MFCC pero usa filtros gammatone que modelan más precisamente la cóclea humana.

### Diferencia con MFCC
- Filtros gammatone: más anchos en altas frecuencias
- Mejor modelado del sistema auditivo humano

### Ventajas
- Más robusto al ruido que MFCC
- Mejor representación de frecuencias altas

### Desventajas
- Menor uso en literatura
- Menos recursos disponibles

### Referencias
- Comparative Analysis of Speaker Identification (2024)
- Gammatone filter bank design

---

## 4. PLP + RASTA

### Descripción
Perceptual Linear Prediction con RASTA (RelAtive SpecTral) filtering.

### Características
- Usa predicción lineal
- Aplica procesamiento perceptual
- RASTA filtra variaciones lentas del espectro

### Ventajas
- Robusto al ruido
- Menos sensible a variaciones de canal

### Desventajas
- Menor popularidad actual
- Puede perder información relevante

### Referencias
- Automatic Speech Recognition Features Extraction Techniques (2021)

---

## 5. SincNet

### Descripción
Arquitectura que aprende filtros adaptados a los datos desde cero, en lugar de usar filtros predefinidos como MFCC.

### Características
- Primera capa: filtros Sinc aprendidos
- Kernel shape: función matemática parametrizada
- Aprende bancos de filtros óptimos para la tarea

### Ventajas
- Filters learned específicos para el dominio
- Puede superar a MFCC en speaker recognition
- Menor sesgo inductivo

### Desventajas
- Requiere más datos
- Mayor tiempo de entrenamiento

### Referencias
- SincNet: Learning speaker filters (Ravanelli et al., 2018)

---

## 6. Multi-Feature Fusion

### Descripción
Combinar múltiples tipos de features para obtener información complementaria.

### Features adicionales a MFCC
- Spectral centroid
- Spectral bandwidth
- Spectral contrast
- Zero crossing rate
- RMS energy
- Pitch/F0
- Chroma features

### Ventajas
- Mayor robustez
- Información complementaria
- Mejor generalización

### Desventajas
- Alta dimensionalidad
- Requiere feature selection
- Mayor complejidad

### Referencias
- Feature selection for emotion recognition in speech (2025)
- Multi-criteria Comparison of ASR Features (2021)

---

## Recomendaciones para el Proyecto

### Fase 1 (Implementado)
- MFCC + SVM/RF/X-Vector/ECAPA-TDNN
- Baseline funcional

### Fase 2 (Recomendado)
- Probar Mel spectrogram + CNN
- Comparar con MFCC

### Fase 3 (Avanzado)
- Multi-feature fusion: MFCC + spectral features
- Feature selection con mutual information

### Comparativa de Complejidad

| Método | Dimensionalidad | Robustez | Complejidad |
|--------|-----------------|----------|-------------|
| MFCC | 13-40 | Baja | Baja |
| GTCC | 13-40 | Media | Media |
| Spectrogram | Alta | Alta | Alta |
| PLP | 13 | Media | Baja |
| Multi-fusion | Alta | Alta | Alta |

### Papers Clave

1. **X-Vectors**: Snyder et al., ICASSP 2018
2. **ECAPA-TDNN**: Desplanques et al., Interspeech 2020
3. **SincNet**: Ravanelli et al., 2018
4. **Comparative Study**: Kurzekar et al., 2014
5. **ASR Features**: Multiple papers, 2021
