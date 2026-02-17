#!/bin/bash
# Entrenar todas las duraciones para k=10 y overlap=0.5 usando X-Vector
# Con cronometraje detallado

K_FOLDS=10
OVERLAP=0.5
DURATIONS=(1 2 5 10 20 30 50)

SCRIPT_START=$(date +%s)

echo "=================================="
echo "ENTRENAMIENTO X-VECTOR CON TIMING"
echo "k=$K_FOLDS, overlap=$OVERLAP"
echo "Inicio: $(date)"
echo "=================================="

for dur in "${DURATIONS[@]}"; do
    DUR_START=$(date +%s)
    echo ""
    echo "========================================"
    echo "Entrenando duración: ${dur}s"
    echo "Inicio: $(date)"
    echo "========================================"

    python entrenar_xvector.py \
        --duration $dur \
        --overlap $OVERLAP \
        --k-folds $K_FOLDS \
        --seed 42

    EXIT_CODE=$?
    DUR_END=$(date +%s)
    DUR_TIME=$((DUR_END - DUR_START))

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR en duración ${dur}s"
    fi

    echo ""
    echo "----------------------------------------"
    echo "Duración ${dur}s completada"
    echo "Tiempo: ${DUR_TIME}s ($((DUR_TIME/60)) min $((DUR_TIME%60)) seg)"
    echo "----------------------------------------"
done

SCRIPT_END=$(date +%s)
TOTAL_TIME=$((SCRIPT_END - SCRIPT_START))

echo ""
echo "=================================="
echo "ENTRENAMIENTO COMPLETADO"
echo "=================================="
echo "Tiempo total: ${TOTAL_TIME}s ($((TOTAL_TIME/60)) min $((TOTAL_TIME%60)) seg)"
echo "Fin: $(date)"
echo "=================================="
