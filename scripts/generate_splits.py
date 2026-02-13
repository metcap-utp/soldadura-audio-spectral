import argparse
from pathlib import Path

from mfcc_baseline.config import load_config
from mfcc_baseline.splits import SplitConfig, generate_splits


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generar splits estratificados por sesion"
    )
    parser.add_argument(
        "--duration", type=int, required=True, choices=[1, 2, 5, 10, 20, 30, 50]
    )
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    split_cfg = SplitConfig(duration=args.duration, overlap_ratio=args.overlap)
    output_dir = Path(args.output_dir)

    generate_splits(
        audio_root=cfg.audio_root,
        cfg=split_cfg,
        output_dir=output_dir,
        sr=cfg.sample_rate,
    )

    print(f"Splits generados en: {output_dir}")


if __name__ == "__main__":
    main()
