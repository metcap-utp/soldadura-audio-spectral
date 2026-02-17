# Instrucciones para Generar Imágenes de Arquitecturas

## Requisitos

- Python 3.8+
- LaTeX (texlive-full o equivalente con tikz)
- PlotNeuralNet instalado en `/home/luis/PlotNeuralNet/`
- pdftoppm (poppler-utils) para conversión a PNG

## Uso

### Generar todos los diagramas

```bash
cd arquitectura/
./compilar_todos.sh
```

### Generar un diagrama específico

```bash
# X-Vector
python xvector_diagram.py
pdflatex -interaction=nonstopmode xvector_diagram.tex
pdftoppm xvector_diagram.pdf xvector_diagram -png -r 200

# ECAPA-TDNN
python ecapa_diagram.py
pdflatex -interaction=nonstopmode ecapa_diagram.tex
pdftoppm ecapa_diagram.pdf ecapa_diagram -png -r 200

# FeedForward
python feedforward_diagram.py
pdflatex -interaction=nonstopmode feedforward_diagram.tex
pdftoppm feedforward_diagram.pdf feedforward_diagram -png -r 200
```

## Convenciones

### Formatos de dimensiones

- **Conv1D**: `out×in×k` (ej: `512×240×5`)
- **FC (Linear)**: `in×out` (ej: `3000×512`)
- **Heads de clasificación**: `emb×clases` (ej: `512×3`)

### Estructura de archivos

Cada modelo tiene:
- `<modelo>_diagram.py`: Script Python generador
- `<modelo>_diagram.tex`: Código LaTeX/TikZ generado
- `<modelo>_diagram.pdf`: Diagrama vectorial
- `<modelo>_diagram-1.png`: Imagen rasterizada (200 DPI)

### Colores usados

- **ConvColor** (naranja): Capas convolucionales TDNN
- **ConvReluColor** (rojo claro): Activaciones (BN + ReLU)
- **FcColor** (azul): Capas fully connected
- **PoolColor** (rojo): Pooling (Stats, Attentive)
- **SoftmaxColor** (magenta): Heads de clasificación

### Tipografía

- Títulos: `\small\centering`
- Dimensiones: `\footnotesize`
- Texto multilínea: Usar `\parbox{<ancho>}{\centering ...}`

## Notas importantes

1. Evitar nodos sueltos - todo texto debe estar dentro de las capas
2. No usar comas en captions dentro de `\pic` (usar `{,}` si es necesario)
3. Para texto multilínea, usar `\parbox` con `\centering`
4. El script genera automáticamente el archivo .tex - no editar manualmente

## Solución de problemas

### PDF en blanco
- Revisar que no haya `\n` dentro de argumentos TikZ
- Verificar que los nombres de colores estén definidos

### Error de pgfkeys
- Falta de `\parbox` para texto largo
- Comas sin escapar en captions

### Texto desordenado
- No usar `\node` externos al sistema de capas
- Mantener todo el texto dentro de los elementos de capa

## Modelos documentados

1. **X-Vector**: TDNN multi-task con Stats Pooling (5 frame-layers + 1 segment)
2. **ECAPA-TDNN**: TDNN con Res2Net blocks, MFA y Attentive Statistics Pooling
3. **FeedForward**: Clasificador feedforward simple para features agregados
