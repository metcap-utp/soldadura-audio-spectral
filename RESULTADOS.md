# Resultados - Spectral Analysis (MFCC)

## Configuración

- **Duración de segmento**: 10 segundos
- **Overlap**: 0.5 (50%)
- **K-Folds**: 10
- **Seed**: 42
- **Features**: MFCC (40 coeficientes)

## Modelos Entrenados

1. **X-Vector**: Red TDNN (Time Delay Neural Network)
2. **ECAPA-TDNN**: ECAPA-TDNN con multi-head attention
3. **FeedForward**: Red neuronal fully-connected

---

## Resultados Test Set (K-Fold Cross-Validation)

| Modelo | Plate | Electrode | Current | **Promedio** |
|--------|-------|-----------|---------|--------------|
| ECAPA-TDNN | 96.59% | 97.68% | 99.45% | **97.91%** |
| X-Vector | 96.89% | 97.28% | 99.45% | 97.87% |
| FeedForward | 95.25% | 96.56% | 99.46% | 97.09% |

**Mejor modelo (Test Set)**: ECAPA-TDNN con 97.91% de accuracy promedio

---

## Resultados Blind Set (Evaluación Ciega)

| Modelo | Plate | Electrode | Current | **Promedio** |
|--------|-------|-----------|---------|--------------|
| ECAPA-TDNN | 98.12% | 96.71% | 99.30% | **98.04%** |
| X-Vector | 96.95% | 96.71% | 98.83% | 97.50% |
| FeedForward | 94.84% | 95.77% | 99.53% | 96.71% |

**Mejor modelo (Blind Set)**: ECAPA-TDNN con 98.04% de accuracy promedio

---

## Métricas Globales (Blind Set)

| Modelo | Exact Match | Hamming Accuracy |
|--------|-------------|------------------|
| **ECAPA-TDNN** | **95.07%** | **98.04%** |
| X-Vector | 94.37% | 97.50% |
| FeedForward | 92.25% | 96.71% |

- **Exact Match**: Porcentaje de muestras donde las 3 etiquetas son correctas simultáneamente
- **Hamming Accuracy**: Promedio de etiquetas correctas (1/3 por muestra)

---

## Análisis

1. **ECAPA-TDNN** es el mejor modelo tanto en Test Set como en Blind Set
2. Las métricas de Blind Set son consistentes con Test Set, indicando buen generalization
3. La tarea **Current** tiene el mayor accuracy (>99%) en todos los modelos
4. La tarea **Plate** es la más difícil, con mayor variación entre modelos

---

## Archivos Generados

### Gráficas
- `10seg/graficas/cv_modelos_10seg.png` - Test Set por modelo
- `10seg/graficas/cv_resumen_10seg.png` - Test Set agrupado
- `10seg/graficas/blind_modelos_10seg.png` - Blind Set por modelo
- `10seg/graficas/blind_resumen_10seg.png` - Blind Set agrupado
- `10seg/graficas/global_metrics_10seg.png` - Métricas globales

### Modelos
- `10seg/modelos/xvector/k10_overlap_0.5/` - 10 folds + SWA
- `10seg/modelos/ecapa/k10_overlap_0.5/` - 10 folds + SWA
- `10seg/modelos/feedforward/k10_overlap_0.5/` - 10 folds + SWA

### Resultados
- `10seg/resultados.json` - Métricas completas en formato JSON
