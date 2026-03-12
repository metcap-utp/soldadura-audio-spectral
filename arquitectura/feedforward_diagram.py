import sys

sys.path.append("/home/luis/PlotNeuralNet/")
from pycore.tikzeng import *


# Capa FC Block personalizada (FC + BN + ReLU + Dropout)
def to_FCBlock(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=3,
    height=30,
    depth=30,
    caption=" ",
):
    return (
        r"""
\pic[shift={"""
        + offset
        + """}] at """
        + to
        + """ 
    {RightBandedBox={
        name="""
        + name
        + """,
        caption="""
        + caption
        + """,
        fill=\FcColor,
        bandfill=\FcReluColor,
        height="""
        + str(height)
        + """,
        width="""
        + str(width)
        + """,
        depth="""
        + str(depth)
        + """
        }
    };
"""
    )


# Capa Clasificador personalizada
def to_Classifier(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=2,
    height=15,
    depth=15,
    caption=" ",
):
    return (
        r"""
\pic[shift={"""
        + offset
        + """}] at """
        + to
        + """ 
    {Box={
        name="""
        + name
        + """,
        caption="""
        + caption
        + """,
        fill=\SoftmaxColor,
        height="""
        + str(height)
        + """,
        width="""
        + str(width)
        + """,
        depth="""
        + str(depth)
        + """
        }
    };
"""
    )


def to_Backbone(name, caption, height=26, depth=26, width=3):
    return (
        r"""
\pic[shift={(0,0,0)}] at (0,0,0)
    {Box={
        name="""
        + name
        + """,
        caption="""
        + caption
        + """,
        fill=\ConvColor,
        height="""
        + str(height)
        + """,
        width="""
        + str(width)
        + """,
        depth="""
        + str(depth)
        + """
        }
    };"""
    )


arch = [
    to_head("/home/luis/PlotNeuralNet/"),
    to_cor(),
    to_begin(),
    # Spectral-MFCC backbone
    to_Backbone(
        name="spectral_mfcc",
        caption=r"{\parbox{2.8cm}{\centering\small\textbf{Spectral-MFCC}\\\footnotesize 1$\times$1$\times$240}}",
        height=30,
        depth=30,
        width=3,
    ),
    # Hidden Layer 1: FC(240, 512) + BN + ReLU + Dropout
    to_FCBlock(
        name="fc1",
        offset="(2.5,0,0)",
        to="(spectral_mfcc-east)",
        width=3,
        height=40,
        depth=40,
        caption=r"""{\parbox{2.5cm}{\centering\small\textbf{FC-1}\\\footnotesize 240$\\times$512\\\footnotesize BN + ReLU}}""",
    ),
    to_connection("spectral_mfcc", "fc1"),
    # Hidden Layer 2: FC(512, 256) + BN + ReLU + Dropout
    to_FCBlock(
        name="fc2",
        offset="(2.5,0,0)",
        to="(fc1-east)",
        width=2.5,
        height=32,
        depth=32,
        caption=r"""{\parbox{2.5cm}{\centering\small\textbf{FC-2}\\\footnotesize 512$\\times$256\\\footnotesize BN + ReLU}}""",
    ),
    to_connection("fc1", "fc2"),
    # Hidden Layer 3: FC(256, 128) + BN + ReLU + Dropout
    to_FCBlock(
        name="fc3",
        offset="(2.5,0,0)",
        to="(fc2-east)",
        width=2,
        height=28,
        depth=28,
        caption=r"""{\parbox{2.5cm}{\centering\small\textbf{FC-3}\\\footnotesize 256$\\times$128\\\footnotesize BN + ReLU}}""",
    ),
    to_connection("fc2", "fc3"),
    # Head: Espesor (128, 3)
    to_Classifier(
        name="head_espesor",
        offset="(3.5,3.5,0)",
        to="(fc3-east)",
        width=2,
        height=12,
        depth=12,
        caption=r"""{\parbox{2.2cm}{\centering\small\textbf{Espesor}\\\footnotesize 128$\\times$3}}""",
    ),
    # Head: Electrodo (128, 4)
    to_Classifier(
        name="head_electrodo",
        offset="(3.5,0,0)",
        to="(fc3-east)",
        width=2,
        height=14,
        depth=14,
        caption=r"""{\parbox{2.2cm}{\centering\small\textbf{Electrodo}\\\footnotesize 128$\\times$4}}""",
    ),
    # Head: Corriente (128, 2)
    to_Classifier(
        name="head_corriente",
        offset="(3.5,-3.5,0)",
        to="(fc3-east)",
        width=2,
        height=10,
        depth=10,
        caption=r"""{\parbox{2.2cm}{\centering\small\textbf{Corriente}\\\footnotesize 128$\\times$2}}""",
    ),
    # Conexiones a los heads
    r"""\draw [connection]  (fc3-east) -- node {\midarrow} (head_espesor-west);""",
    r"""\draw [connection]  (fc3-east) -- node {\midarrow} (head_electrodo-west);""",
    r"""\draw [connection]  (fc3-east) -- node {\midarrow} (head_corriente-west);""",
    to_end(),
]


def main():
    namefile = str(sys.argv[0]).split(".")[0]
    to_generate(arch, namefile + ".tex")
    print(f"Archivo generado: {namefile}.tex")


if __name__ == "__main__":
    main()
