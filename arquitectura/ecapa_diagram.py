import sys

sys.path.append("/home/luis/PlotNeuralNet/")
from pycore.tikzeng import *


# Capa TDNN personalizada
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


# Capa Res2Net personalizada
def to_Res2Net(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=3,
    height=35,
    depth=35,
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


# Capa MFA personalizada
def to_MFA(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=3,
    height=40,
    depth=40,
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


# Capa ASP personalizada
def to_ASP(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=2,
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


def to_Head(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=2.5,
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
    };"""
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
        caption=r"{\parbox{2.8cm}{\centering\small\textbf{Spectral-MFCC}\\\footnotesize 1$\times$1$\times$40}}",
        height=30,
        depth=30,
        width=3,
    ),
    # Layer 1: TDNN Block
    to_TDNN(
        name="layer1",
        offset="(2.5,0,0)",
        to="(spectral_mfcc-east)",
        width=3,
        height=35,
        depth=35,
        caption="""{\\parbox{2.8cm}{\\centering\\small\\textbf{TDNN-1}\\\\\\footnotesize 512$\\times$40$\\times$5}}""",
    ),
    to_connection("spectral_mfcc", "layer1"),
    # Res2Net Block 1
    to_Res2Net(
        name="res2net1",
        offset="(2.5,0,0)",
        to="(layer1-east)",
        width=3,
        height=35,
        depth=35,
        caption="""{\\parbox{2.8cm}{\\centering\\small\\textbf{Res2Net-1}\\\\\\footnotesize 512$\\times$512$\\times$3\\\\s=8}}""",
    ),
    to_connection("layer1", "res2net1"),
    # Res2Net Block 2
    to_Res2Net(
        name="res2net2",
        offset="(2.5,0,0)",
        to="(res2net1-east)",
        width=3,
        height=35,
        depth=35,
        caption="""{\\parbox{2.8cm}{\\centering\\small\\textbf{Res2Net-2}\\\\\\footnotesize 512$\\times$512$\\times$3\\\\s=8}}""",
    ),
    to_connection("res2net1", "res2net2"),
    # Res2Net Block 3
    to_Res2Net(
        name="res2net3",
        offset="(2.5,0,0)",
        to="(res2net2-east)",
        width=3,
        height=35,
        depth=35,
        caption="""{\\parbox{2.8cm}{\\centering\\small\\textbf{Res2Net-3}\\\\\\footnotesize 512$\\times$512$\\times$3\\\\s=8}}""",
    ),
    to_connection("res2net2", "res2net3"),
    # MFA
    to_MFA(
        name="mfa",
        offset="(3,0,0)",
        to="(res2net3-east)",
        width=4,
        height=40,
        depth=40,
        caption="""{\\parbox{3cm}{\\centering\\small\\textbf{MFA}\\\\\\footnotesize Conv 1$\\times$1\\\\Multi-Feature\\\\Aggregation}}""",
    ),
    to_connection("res2net3", "mfa"),
    # ASP
    to_ASP(
        name="asp",
        offset="(2.8,0,0)",
        to="(mfa-east)",
        width=3,
        height=35,
        depth=35,
        caption="""{\\parbox{2.8cm}{\\centering\\small\\textbf{ASP}\\\\\\footnotesize Attentive Stats\\\\Pooling\\\\3072}}""",
    ),
    to_connection("mfa", "asp"),
    # FC
    to_FC(
        name="fc",
        offset="(2.5,0,0)",
        to="(asp-east)",
        width=2,
        height=25,
        depth=25,
        caption="""{\\parbox{2.2cm}{\\centering\\small\\textbf{FC}\\\\\\footnotesize 3072$\\times$192}}""",
    ),
    to_connection("asp", "fc"),
    # Head: Espesor (192, 3)
    to_Head(
        name="head_espesor",
        offset="(3.5,3.5,0)",
        to="(fc-east)",
        width=2.5,
        height=12,
        depth=12,
        caption="""{\\parbox{2.2cm}{\\centering\\small\\textbf{Espesor}\\\\\\footnotesize 192$\\times$3}}""",
    ),
    # Head: Electrodo (192, 4)
    to_Head(
        name="head_electrodo",
        offset="(3.5,0,0)",
        to="(fc-east)",
        width=2.5,
        height=14,
        depth=14,
        caption="""{\\parbox{2.2cm}{\\centering\\small\\textbf{Electrodo}\\\\\\footnotesize 192$\\times$4}}""",
    ),
    # Head: Corriente (192, 2)
    to_Head(
        name="head_corriente",
        offset="(3.5,-3.5,0)",
        to="(fc-east)",
        width=2.5,
        height=10,
        depth=10,
        caption="""{\\parbox{2.2cm}{\\centering\\small\\textbf{Corriente}\\\\\\footnotesize 192$\\times$2}}""",
    ),
    # Conexiones a los heads
    r"""\draw [connection]  (fc-east) -- node {\midarrow} (head_espesor-west);""",
    r"""\draw [connection]  (fc-east) -- node {\midarrow} (head_electrodo-west);""",
    r"""\draw [connection]  (fc-east) -- node {\midarrow} (head_corriente-west);""",
    to_end(),
]


def main():
    namefile = str(sys.argv[0]).split(".")[0]
    to_generate(arch, namefile + ".tex")
    print(f"Archivo generado: {namefile}.tex")


if __name__ == "__main__":
    main()
