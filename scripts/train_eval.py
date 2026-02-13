import argparse
from pathlib import Path

from mfcc_baseline.config import load_config
from mfcc_baseline.dataset import FeatureConfig
from mfcc_baseline.train import train_and_eval


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenar y evaluar baseline MFCC")
    parser.add_argument(
        "--duration", type=int, required=True, choices=[1, 2, 5, 10, 20, 30, 50]
    )
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--splits-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="svm", choices=["svm", "rf"])
    parser.add_argument("--n-mfcc", type=int, default=40)
    parser.add_argument("--no-deltas", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)

    feature_cfg = FeatureConfig(n_mfcc=args.n_mfcc, include_deltas=not args.no_deltas)

    train_and_eval(
        train_csv=splits_dir / "train.csv",
        test_csv=splits_dir / "test.csv",
        audio_root=cfg.audio_root,
        segment_duration=args.duration,
        overlap_ratio=args.overlap,
        sample_rate=cfg.sample_rate,
        feature_cfg=feature_cfg,
        cache_dir=cfg.cache_dir,
        output_dir=output_dir,
        model_type=args.model,
    )

    print(f"Resultados guardados en: {output_dir}")


if __name__ == "__main__":
    main()
