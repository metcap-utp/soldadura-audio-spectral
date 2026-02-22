#!/bin/bash
# Entrenar todos los modelos para todas las duraciones (SECUENCIAL)
# Con cronometraje detallado para comparación de tiempos
# Uso: ./entrenar_todos.sh

K_FOLDS=10
OVERLAP=0.5
DURATIONS=(1 2 5 10 20 30 50)
MODELS=("xvector" "ecapa" "feedforward")

SCRIPT_START=$(date +%s)
declare -A TIMES

TOTAL=$((${#DURATIONS[@]} * ${#MODELS[@]}))
COUNT=0

echo "========================================"
echo "ENTRENAMIENTO SECUENCIAL DE MODELOS"
echo "========================================"
echo "K-Folds: $K_FOLDS"
echo "Overlap: $OVERLAP"
echo "Duraciones: ${DURATIONS[@]}"
echo "Modelos: ${MODELS[@]}"
echo "Total entrenamientos: $TOTAL"
echo "Inicio: $(date)"
echo "========================================"

for dur in "${DURATIONS[@]}"; do
    echo ""
    echo "========================================"
    echo "DURACIÓN: ${dur}s"
    echo "========================================"
    
    for model in "${MODELS[@]}"; do
        COUNT=$((COUNT + 1))
        MODEL_START=$(date +%s)
        
        echo ""
        echo "[$COUNT/$TOTAL] Entrenando: $model - ${dur}s"
        echo "Inicio: $(date +%H:%M:%S)"
        echo "----------------------------------------"
        
        python "entrenar_${model}.py" \
            --duration $dur \
            --overlap $OVERLAP \
            --k-folds $K_FOLDS \
            --seed 42
        
        EXIT_CODE=$?
        MODEL_END=$(date +%s)
        MODEL_TIME=$((MODEL_END - MODEL_START))
        
        TIMES["${model}_${dur}s"]=${MODEL_TIME}
        
        if [ $EXIT_CODE -ne 0 ]; then
            echo "ERROR en $model - ${dur}s"
        fi
        
        echo ""
        echo "Completado: ${model} - ${dur}s"
        echo "Tiempo: ${MODEL_TIME}s ($((MODEL_TIME/60)) min $((MODEL_TIME%60)) seg)"
        echo "----------------------------------------"
    done
done

SCRIPT_END=$(date +%s)
TOTAL_TIME=$((SCRIPT_END - SCRIPT_START))

echo ""
echo "========================================"
echo "RESUMEN DE TIEMPOS"
echo "========================================"
echo ""
printf "%-12s %-12s %-15s\n" "Modelo" "Duración" "Tiempo"
printf "%-12s %-12s %-15s\n" "------" "--------" "------"

for dur in "${DURATIONS[@]}"; do
    for model in "${MODELS[@]}"; do
        key="${model}_${dur}s"
        t=${TIMES[$key]:-0}
        mins=$((t/60))
        secs=$((t%60))
        printf "%-12s %-12s %3d min %2d seg\n" "$model" "${dur}s" "$mins" "$secs"
    done
    echo ""
done

echo "========================================"
echo "TIEMPO TOTAL: ${TOTAL_TIME}s ($((TOTAL_TIME/60)) min $((TOTAL_TIME%60)) seg)"
echo "Fin: $(date)"
echo "========================================"
