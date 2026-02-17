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


arch = [
    to_head("/home/luis/PlotNeuralNet/"),
    to_cor(),
    to_begin(),
    
    # Input - Aggregated MFCC features
    to_input("input", to="(0,0,0)", width=10, height=10, name="input"),
    
    # Hidden Layer 1: FC(240, 512) + BN + ReLU + Dropout
    to_FCBlock(
        name="fc1",
        offset="(2.5,0,0)",
        to="(input-east)",
        width=3,
        height=40,
        depth=40,
        caption=r"""{\parbox{2.5cm}{\centering\small\textbf{FC-1}\footnotesize 240$\\times$512\\BN + ReLU}}""",
    ),
    to_connection("input", "fc1"),
    
    # Hidden Layer 2: FC(512, 256) + BN + ReLU + Dropout
    to_FCBlock(
        name="fc2",
        offset="(2.5,0,0)",
        to="(fc1-east)",
        width=2.5,
        height=32,
        depth=32,
        caption=r"""{\parbox{2.5cm}{\centering\small\textbf{FC-2}\footnotesize 512$\\times$256\\BN + ReLU}}""",
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
        caption=r"""{\parbox{2.5cm}{\centering\small\textbf{FC-3}\footnotesize 256$\\times$128\\BN + ReLU}}""",
    ),
    to_connection("fc2", "fc3"),
    
    # Classifier: FC(128, C)
    to_Classifier(
        name="classifier",
        offset="(2.5,0,0)",
        to="(fc3-east)",
        width=2,
        height=18,
        depth=18,
        caption=r"""{\parbox{2cm}{\centering\small\textbf{Clasificador}\footnotesize 128$\\times$C}}""",
    ),
    to_connection("fc3", "classifier"),
    
    to_end(),
]


def main():
    namefile = str(sys.argv[0]).split(".")[0]
    to_generate(arch, namefile + ".tex")
    print(f"Archivo generado: {namefile}.tex")


if __name__ == "__main__":
    main()
