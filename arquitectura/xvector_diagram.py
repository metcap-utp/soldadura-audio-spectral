import sys

sys.path.append("/home/luis/PlotNeuralNet/")
from pycore.tikzeng import *


# Capa TDNN personalizada (Conv1D + BN + ReLU)
def to_TDNN(
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
        fill=\ConvColor,
        bandfill=\ConvReluColor,
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


# Capa FC personalizada
def to_FC(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=2,
    height=20,
    depth=20,
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
        fill=\FcColor,
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


# Capa Stats Pooling personalizada
def to_StatsPool(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=2,
    height=25,
    depth=25,
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
        fill=\PoolColor,
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


# Capa Head personalizada
def to_Head(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=2,
    height=12,
    depth=12,
    opacity=0.9,
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
        opacity="""
        + str(opacity)
        + """,
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
    
    # Input - MFCC features
    to_input("input", to="(0,0,0)", width=8, height=8, name="input"),
    
    # Frame 1: Conv1D(240, 512, k=5, d=1) + BN + ReLU
    to_TDNN(
        name="frame1",
        offset="(2.5,0,0)",
        to="(input-east)",
        width=3,
        height=35,
        depth=35,
        caption="""{\\parbox{2.8cm}{\\centering\\small\\textbf{Frame 1}\\\\\\footnotesize 512$\\times$240$\\times$5\\\\ d=1}}""",
    ),
    to_connection("input", "frame1"),
    
    # Frame 2: Conv1D(512, 512, k=3, d=2) + BN + ReLU
    to_TDNN(
        name="frame2",
        offset="(2.5,0,0)",
        to="(frame1-east)",
        width=3,
        height=35,
        depth=35,
        caption="""{\\parbox{2.8cm}{\\centering\\small\\textbf{Frame 2}\\\\\\footnotesize 512$\\times$512$\\times$3\\\\ d=2}}""",
    ),
    to_connection("frame1", "frame2"),
    
    # Frame 3: Conv1D(512, 512, k=3, d=3) + BN + ReLU
    to_TDNN(
        name="frame3",
        offset="(2.5,0,0)",
        to="(frame2-east)",
        width=3,
        height=35,
        depth=35,
        caption="""{\\parbox{2.8cm}{\\centering\\small\\textbf{Frame 3}\\\\\\footnotesize 512$\\times$512$\\times$3\\\\ d=3}}""",
    ),
    to_connection("frame2", "frame3"),
    
    # Frame 4: Conv1D(512, 512, k=1, d=1) + BN + ReLU
    to_TDNN(
        name="frame4",
        offset="(2.5,0,0)",
        to="(frame3-east)",
        width=2.5,
        height=32,
        depth=32,
        caption="""{\\parbox{2.5cm}{\\centering\\small\\textbf{Frame 4}\\\\\\footnotesize 512$\\times$512$\\times$1}}""",
    ),
    to_connection("frame3", "frame4"),
    
    # Frame 5: Conv1D(512, 1500, k=1, d=1) + BN + ReLU
    to_TDNN(
        name="frame5",
        offset="(2.5,0,0)",
        to="(frame4-east)",
        width=4,
        height=40,
        depth=40,
        caption="""{\\parbox{2.8cm}{\\centering\\small\\textbf{Frame 5}\\\\\\footnotesize 1500$\\times$512$\\times$1}}""",
    ),
    to_connection("frame4", "frame5"),
    
    # Stats Pooling: mean + std -> 3000
    to_StatsPool(
        name="stats",
        offset="(2.8,0,0)",
        to="(frame5-east)",
        width=3,
        height=30,
        depth=30,
        caption="""{\\parbox{2.5cm}{\\centering\\small\\textbf{Stats Pool}\\\\\\footnotesize mean + std\\\\3000}}""",
    ),
    to_connection("frame5", "stats"),
    
    # Segment: FC(3000, 512) + BN + ReLU
    to_FC(
        name="segment",
        offset="(2.8,0,0)",
        to="(stats-east)",
        width=2.5,
        height=28,
        depth=28,
        caption="""{\\parbox{2.4cm}{\\centering\\small\\textbf{Segment}\\\\\\footnotesize FC + ReLU\\\\3000$\\times$512}}""",
    ),
    to_connection("stats", "segment"),
    
    # Head: Espesor (512, 3)
    to_Head(
        name="head_espesor",
        offset="(3.5,3.5,0)",
        to="(segment-east)",
        width=2.5,
        height=12,
        depth=12,
        caption="""{\\parbox{2.2cm}{\\centering\\small\\textbf{Espesor}\\\\\\footnotesize 512$\\times$3}}""",
    ),
    
    # Head: Electrodo (512, 4)
    to_Head(
        name="head_electrodo",
        offset="(3.5,0,0)",
        to="(segment-east)",
        width=2.5,
        height=14,
        depth=14,
        caption="""{\\parbox{2.2cm}{\\centering\\small\\textbf{Electrodo}\\\\\\footnotesize 512$\\times$4}}""",
    ),
    
    # Head: Corriente (512, 2)
    to_Head(
        name="head_corriente",
        offset="(3.5,-3.5,0)",
        to="(segment-east)",
        width=2.5,
        height=10,
        depth=10,
        caption="""{\\parbox{2.2cm}{\\centering\\small\\textbf{Corriente}\\\\\\footnotesize 512$\\times$2}}""",
    ),
    
    # Conexiones a los heads
    r"""\draw [connection]  (segment-east) -- node {\midarrow} (head_espesor-west);""",
    r"""\draw [connection]  (segment-east) -- node {\midarrow} (head_electrodo-west);""",
    r"""\draw [connection]  (segment-east) -- node {\midarrow} (head_corriente-west);""",
    
    to_end(),
]


def main():
    namefile = str(sys.argv[0]).split(".")[0]
    to_generate(arch, namefile + ".tex")
    print(f"Archivo generado: {namefile}.tex")


if __name__ == "__main__":
    main()
