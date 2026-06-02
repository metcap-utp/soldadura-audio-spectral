#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "=== Clasificadores Spectral-Analysis ==="
for script in xvector_diagram.py ecapa_diagram.py feedforward_diagram.py; do
    base="${script%.py}"
    echo "--- $base ---"
    python "$script"
    for lang in esp eng; do
        f="${base}_${lang}"
        pdflatex -interaction=nonstopmode "${f}.tex" > "${f}.compile.log" 2>&1 \
            && pdftoppm "${f}.pdf" "${f}" -png -r 300 \
            && echo "  OK: ${f}-1.png" \
            || echo "  ERROR: ${f} — ver ${f}.compile.log"
    done
done
echo "Listo."
