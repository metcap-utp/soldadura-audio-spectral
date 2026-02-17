#!/bin/bash

# Script para compilar todos los diagramas de arquitectura
# Genera archivos .tex, .pdf y .png

set -e

echo "=================================="
echo "Compilando diagramas de arquitectura"
echo "=================================="

# Lista de modelos a compilar
MODELS=("xvector" "ecapa" "feedforward")

for model in "${MODELS[@]}"; do
    echo ""
    echo "Procesando: $model"
    echo "-----------------------------------"
    
    # Generar archivo .tex desde Python
    echo "Generando $model.tex..."
    python "${model}_diagram.py"
    
    # Compilar a PDF
    echo "Compilando a PDF..."
    pdflatex -interaction=nonstopmode "${model}_diagram.tex" > "${model}_compile.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ PDF generado: ${model}_diagram.pdf"
    else
        echo "✗ Error al compilar PDF. Ver ${model}_compile.log"
        continue
    fi
    
    # Convertir a PNG
    echo "Convirtiendo a PNG..."
    pdftoppm "${model}_diagram.pdf" "${model}_diagram" -png -r 200
    
    if [ $? -eq 0 ]; then
        echo "✓ PNG generado: ${model}_diagram-1.png"
    else
        echo "✗ Error al convertir a PNG"
    fi
    
    # Limpiar archivos temporales
    rm -f "${model}_diagram.aux" "${model}_diagram.log"
    
done

echo ""
echo "=================================="
echo "Compilación completada"
echo "=================================="
echo ""
echo "Archivos generados:"
for model in "${MODELS[@]}"; do
    if [ -f "${model}_diagram.pdf" ]; then
        echo "  ✓ ${model}_diagram.pdf"
    fi
    if [ -f "${model}_diagram-1.png" ]; then
        echo "  ✓ ${model}_diagram-1.png"
    fi
done
