# Spectral Analysis - Resultados

## Configuración
- **Duración:** 5 segundos
- **K-folds:** 10
- **Overlap:** 0.5

---

## Métricas por Modelo (Blind Set)

| Modelo | Plate Acc | Electrode Acc | Current Acc | Exact Match | Hamming |
|--------|-----------|---------------|-------------|-------------|---------|
| xvector | 0.9616 | 0.9682 | 0.9934 | 0.9374 | 0.9744 |
| **ecapa** | 0.9682 | 0.9759 | 0.9923 | **0.9451** | **0.9788** |
| feedforward | 0.9429 | 0.9539 | 0.9879 | 0.9111 | 0.9616 |

---

## Mejor Modelo: ecapa

| Métrica | Valor |
|---------|-------|
| Exact Match | **0.9451** |
| Hamming Accuracy | **0.9788** |
| Plate Accuracy | 0.9682 |
| Electrode Accuracy | 0.9759 |
| Current Accuracy | 0.9923 |

---

## Figuras

### Accuracy por Duración
![Accuracy por duración](graficas/accuracy_duracion_blind_set.png)

### Métricas Globales
![Métricas globales](graficas/metricas_globales_blind_set.png)

### Comparación de Modelos
![Backbones](graficas/backbones_blind_set.png)

### Matriz de Confusión - ecapa (Plate)
![Matriz ecapa plate](graficas/matriz_confusion_spectral_ecapa_plate.png)

### Matriz de Confusión - ecapa (Electrode)
![Matriz ecapa electrode](graficas/matriz_confusion_spectral_ecapa_electrode.png)

### Matriz de Confusión - ecapa (Current)
![Matriz ecapa current](graficas/matriz_confusion_spectral_ecapa_current.png)

---

## Conclusiones

1. **ecapa** es el mejor modelo con 94.51% exact match
2. **Current** es la tarea más fácil (>99% accuracy)
3. **Plate** es la más difícil pero sigue siendo muy alta (~97%)
4. Spectral supera significativamente a VGGish y YAMNet
