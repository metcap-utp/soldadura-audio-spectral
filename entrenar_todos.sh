#!/bin/bash
# Entrenar todos los modelos para todas las duraciones
# Uso: ./entrenar_todos.sh [k_folds] [overlap]
# Ejemplo: ./entrenar_todos.sh 10 0.5

K_FOLDS=${1:-10}
OVERLAP=${2:-0.5}

echo "=================================="
echo "ENTRENAMIENTO MASIVO DE MODELOS"
echo "=================================="
echo "K-Folds: $K_FOLDS"
echo "Overlap: $OVERLAP"
echo "=================================="

DURATIONS=(1 2 5 10 20 30 50)
MODELS=("xvector" "ecapa" "feedforward")

TOTAL=$((${#DURATIONS[@]} * ${#MODELS[@]}))
COUNT=0

for model in "${MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "MODELO: $model"
    echo "========================================"
    
    for dur in "${DURATIONS[@]}"; do
        COUNT=$((COUNT + 1))
        echo ""
        echo "[$COUNT/$TOTAL] Entrenando: $model - ${dur}s"
        echo "----------------------------------------"
        
        python "entrenar_${model}.py" \
            --duration $dur \
            --overlap $OVERLAP \
            --k-folds $K_FOLDS \
            --seed 42
        
        if [ $? -ne 0 ]; then
            echo "ERROR en $model - ${dur}s"
        fi
    done
done

echo ""
echo "=================================="
echo "ENTRENAMIENTO COMPLETADO"
echo "=================================="
echo "Total modelos entrenados: $COUNT"
