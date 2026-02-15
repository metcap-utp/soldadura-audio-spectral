#!/bin/bash
# Entrenar modelos apropiados para MFCC (FeedForward y ECAPA)
# XVector no es adecuado para features MFCC agregadas

K_FOLDS=10
OVERLAP=0.5
DURATIONS=(10 20 30 50)  # Solo duraciones mayores para empezar

echo "========================================"
echo "ENTRENAMIENTO MFCC - MODELOS ADECUADOS"
echo "========================================"
echo "K-Folds: $K_FOLDS"
echo "Overlap: $OVERLAP"
echo "Duraciones: ${DURATIONS[@]}"
echo "Modelos: FeedForward, ECAPA-TDNN"
echo "========================================"

for dur in "${DURATIONS[@]}"; do
    echo ""
    echo "========================================"
    echo "DURACIÓN: ${dur}s"
    echo "========================================"
    
    # FeedForward (mejor para MFCC agregados)
    echo ""
    echo "Entrenando FeedForward..."
    python entrenar_feedforward.py --duration $dur --overlap $OVERLAP --k-folds $K_FOLDS --seed 42 2>&1 | tee "${dur}seg/log_feedforward.txt" || echo "Error en FeedForward ${dur}s"
    
    # ECAPA-TDNN (usa MFCC raw)
    echo ""
    echo "Entrenando ECAPA-TDNN..."
    python entrenar_ecapa.py --duration $dur --overlap $OVERLAP --k-folds $K_FOLDS --seed 42 2>&1 | tee "${dur}seg/log_ecapa.txt" || echo "Error en ECAPA ${dur}s"
done

echo ""
echo "========================================"
echo "GENERANDO GRÁFICAS"
echo "========================================"

python scripts/graficar_duraciones.py --k-folds $K_FOLDS --save 2>&1 | tee graficas_duraciones.log || true
python scripts/graficar_overlap.py --k-folds $K_FOLDS --save 2>&1 | tee graficas_overlap.log || true

echo ""
echo "========================================"
echo "ENTRENAMIENTO COMPLETADO"
echo "========================================"
