import sys

sys.path.append("/home/luis/PlotNeuralNet/")
from pycore.tikzeng import *


def lcap(name, dims, w="2.2"):
    return (r"{\parbox{" + w + r"cm}{\centering\large\textbf{" + name
            + r"}\\\normalsize\textbf{" + dims + "}}}")


def to_FCBlock(name, offset="(0,0,0)", to="(0,0,0)", width=3, height=30, depth=30, caption=" "):
    return (r"""
\pic[shift={""" + offset + """}] at """ + to + """
    {RightBandedBox={
        name=""" + name + """,
        caption=""" + caption + """,
        fill=\FcColor,
        bandfill=\FcReluColor,
        height=""" + str(height) + """,
        width=""" + str(width) + """,
        depth=""" + str(depth) + """
        }
    };
""")


def to_Classifier(name, offset="(0,0,0)", to="(0,0,0)", width=2, height=15, depth=15, caption=" "):
    return (r"""
\pic[shift={""" + offset + """}] at """ + to + """
    {Box={
        name=""" + name + """,
        caption=""" + caption + """,
        fill=\SoftmaxColor,
        opacity=0.9,
        height=""" + str(height) + """,
        width=""" + str(width) + """,
        depth=""" + str(depth) + """
        }
    };
""")


def to_Backbone(name, caption, height=26, depth=26, width=3):
    return (r"""
\pic[shift={(0,0,0)}] at (0,0,0)
    {Box={
        name=""" + name + """,
        caption=""" + caption + """,
        fill=\ConvColor,
        height=""" + str(height) + """,
        width=""" + str(width) + """,
        depth=""" + str(depth) + """
        }
    };""")


COLORDEFS = r"""
\usetikzlibrary{calc}
\definecolor{LegConv}{RGB}{255,204,102}
\definecolor{LegConvRelu}{RGB}{255,170,85}
\definecolor{LegPool}{RGB}{196,0,0}
\definecolor{LegFc}{RGB}{153,102,204}
\definecolor{LegFcRelu}{RGB}{164,73,164}
\definecolor{LegSoftmax}{RGB}{106,0,106}
\definecolor{LegEdge}{RGB}{32,128,128}
"""

LEGEND_ENG = r"""
\path (current bounding box.south) coordinate (BBOXS);
\begin{scope}[shift={($(BBOXS)+(-9.8,-3.5)$)}]
  \node[anchor=west,font=\Large\bfseries] at (0,2.0) {Legend};
  \fill[LegConv] (0,0) rectangle (2.0,0.55);
  \node[anchor=west,font=\Large] at (2.4,0.275) {Pre-trained extractor};
  \fill[LegFc] (0,-0.9) rectangle (1.7,-0.35); \fill[LegFcRelu] (1.7,-0.9) rectangle (2.0,-0.35);
  \node[anchor=west,font=\Large] at (2.4,-0.625) {FC + BN + ReLU + Dropout};
  \fill[LegSoftmax,opacity=0.9] (0,-1.8) rectangle (2.0,-1.25);
  \node[anchor=west,font=\Large] at (2.4,-1.525) {Classification (heads)};
  \draw[-Stealth,line width=1pt,LegEdge] (0,-2.6) -- (2.0,-2.6);
  \node[anchor=west,font=\Large] at (2.4,-2.6) {Data flow};
  \draw[rounded corners=4pt,black,line width=0.6pt] (-0.4,2.5) rectangle (20,-3.2);
\end{scope}
"""

arch = [
    to_head("/home/luis/PlotNeuralNet/"),
    to_cor(),
    COLORDEFS,
    to_begin(),
    to_Backbone("spectral_mfcc",
                lcap("MFCC", r"1$\times$1$\times$240"),
                height=30, depth=30, width=3),
    to_FCBlock("fc1", "(2.6,0,0)", "(spectral_mfcc-east)", 3, 40, 40,
               lcap("FC-1", r"240$\times$512")),
    to_connection("spectral_mfcc", "fc1"),
    to_FCBlock("fc2", "(2.6,0,0)", "(fc1-east)", 2.5, 32, 32,
               lcap("FC-2", r"512$\times$256")),
    to_connection("fc1", "fc2"),
    to_FCBlock("fc3", "(2.6,0,0)", "(fc2-east)", 2, 28, 28,
               lcap("FC-3", r"256$\times$128")),
    to_connection("fc2", "fc3"),
    to_Classifier("head_espesor", "(3.5,4.5,0)", "(fc3-east)", 2, 12, 12, " "),
    to_Classifier("head_electrodo", "(3.5,0,0)", "(fc3-east)", 2, 14, 14, " "),
    to_Classifier("head_corriente", "(3.5,-4.5,0)", "(fc3-east)", 2, 10, 10, " "),
    r"""\draw [connection]  (fc3-east) -- node {\midarrow} (head_espesor-west);""",
    r"""\draw [connection]  (fc3-east) -- node {\midarrow} (head_electrodo-west);""",
    r"""\draw [connection]  (fc3-east) -- node {\midarrow} (head_corriente-west);""",
        r"""\path (current bounding box.east) coordinate (BB-EAST);
""",
    r"""\node[anchor=west, xshift=8pt, align=center] at (BB-EAST |- head_espesor-east) {""" + lcap("Espesor", r"128$\times$3", "2.0") + r"""};""",
    r"""\node[anchor=west, xshift=8pt, align=center] at (BB-EAST |- head_electrodo-east) {""" + lcap("Electrodo", r"128$\times$4", "2.0") + r"""};""",
    r"""\node[anchor=west, xshift=8pt, align=center] at (BB-EAST |- head_corriente-east) {""" + lcap("Corriente", r"128$\times$2", "2.0") + r"""};""",
    r"""
\path (current bounding box.south) coordinate (BBOXS);
\begin{scope}[shift={($(BBOXS)+(-9.8,-3.5)$)}]
  \node[anchor=west,font=\Large\bfseries] at (0,2.0) {Leyenda};
  \fill[LegConv] (0,0) rectangle (2.0,0.55);
  \node[anchor=west,font=\Large] at (2.4,0.275) {Extractor pre-entrenado};
  \fill[LegFc] (0,-0.9) rectangle (1.7,-0.35); \fill[LegFcRelu] (1.7,-0.9) rectangle (2.0,-0.35);
  \node[anchor=west,font=\Large] at (2.4,-0.625) {FC + BN + ReLU + Dropout};
  \fill[LegSoftmax,opacity=0.9] (0,-1.8) rectangle (2.0,-1.25);
  \node[anchor=west,font=\Large] at (2.4,-1.525) {Clasificacion (cabezas)};
  \draw[-Stealth,line width=1pt,LegEdge] (0,-2.6) -- (2.0,-2.6);
  \node[anchor=west,font=\Large] at (2.4,-2.6) {Flujo de datos};
  \draw[rounded corners=4pt,black,line width=0.6pt] (-0.4,2.5) rectangle (20,-3.2);
\end{scope}
""",
    to_end(),
]


def main():
    base = str(sys.argv[0]).split(".")[0]
    arch_esp = arch[:-1] + [to_end()]
    to_generate(arch_esp, base + "_esp.tex")
    print(f"Generado: {base}_esp.tex")
    arch_eng = arch[:-2] + [LEGEND_ENG, to_end()]
    to_generate(arch_eng, base + "_eng.tex")
    print(f"Generado: {base}_eng.tex")


if __name__ == "__main__":
    main()
