#!/bin/bash
# Script completo: entrenar todos los modelos + generar gráficas
# Uso: ./ejecutar_completo.sh

set -e  # Detener si hay errores

K_FOLDS=10
OVERLAP=0.5
DURATIONS=(1 2 5 10 20 30 50)

echo "========================================"
echo "ENTRENAMIENTO Y EVALUACIÓN COMPLETA"
echo "========================================"
echo "K-Folds: $K_FOLDS"
echo "Overlap: $OVERLAP"
echo "Duraciones: ${DURATIONS[@]}"
echo "========================================"

# Función para entrenar un modelo
train_model() {
    local model=$1
    local duration=$2
    
    echo ""
    echo "========================================"
    echo "Entrenando: $model - ${duration}s"
    echo "========================================"
    
    python "entrenar_${model}.py" \
        --duration $duration \
        --overlap $OVERLAP \
        --k-folds $K_FOLDS \
        --seed 42 \
        2>&1 | tee "${duration}seg/logs_${model}.txt"
    
    if [ $? -eq 0 ]; then
        echo "✓ $model - ${duration}s completado"
    else
        echo "✗ ERROR en $model - ${duration}s"
    fi
}

# Crear directorios de logs
for dur in "${DURATIONS[@]}"; do
    mkdir -p "${dur}seg"
done

# Entrenar todos los modelos
for dur in "${DURATIONS[@]}"; do
    for model in xvector ecapa feedforward; do
        train_model $model $dur
    done
done

echo ""
echo "========================================"
echo "GENERANDO GRÁFICAS"
echo "========================================"

# Generar gráficas
python scripts/graficar_duraciones.py --k-folds $K_FOLDS --save 2>&1 | tee graficas_duraciones.log || true
python scripts/graficar_overlap.py --k-folds $K_FOLDS --save 2>&1 | tee graficas_overlap.log || true

echo ""
echo "========================================"
echo "PROCESO COMPLETADO"
echo "========================================"
echo "Revisa los archivos:"
echo "  - {N}seg/resultados.json"
echo "  - graficas/*.png"
