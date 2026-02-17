# Soldadura Audio MFCC - Spectral Analysis

Clasificación de audio de soldadura SMAW usando features MFCC y redes neuronales profundas.

## Descripción

Este proyecto implementa un pipeline de clasificación multi-task para identificar:
- Espesor de placa (3 clases)
- Tipo de electrodo (4 clases)  
- Tipo de corriente (2 clases)

A partir de segmentos de audio de soldadura. Utiliza características MFCC (40 coeficientes) 
extraídas directamente del audio, combinadas con arquitecturas de deep learning:
X-Vector (TDNN), ECAPA-TDNN y FeedForward.

## Características

- Extracción de features MFCC con cache
- K-Fold Cross-Validation (10 folds)
- Ensemble voting para evaluación
- Métricas globales (Exact Match, Hamming Accuracy)
- Comparación justa con pipeline VGGish

## Uso

```bash
# Generar splits
python generar_splits.py --duration 10 --overlap 0.5

# Entrenar modelo
python entrenar_ecapa.py --duration 10 --overlap 0.5 --k-folds 10

# Comparar resultados
python scripts/comparar_modelos.py 10seg --save
```

## Comparación con VGGish

Este proyecto está diseñado para comparación directa con el proyecto vggish-backbone.
Ambos usan la misma estructura, splits y configuración para permitir evaluación 
justa entre MFCC features vs VGGish embeddings.
