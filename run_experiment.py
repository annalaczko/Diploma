"""Run one experiment from a config file (see configs/default.yaml for every
setting and pipeline/experiment.py for what a run records):

    python run_experiment.py configs/vae_k16_beta32.yaml
    python run_experiment.py configs/vae_k16_beta32.yaml --set clustering.cluster_num=30
    python run_experiment.py configs/vae_k16_beta32.yaml --dry-run
"""

import argparse

from pipeline.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a pipeline experiment from a YAML config")
    parser.add_argument("config", nargs="?", default=None,
                        help="experiment config, merged over configs/default.yaml (omit to run the defaults)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                        help="override a setting, e.g. --set model.params.k=32 (repeatable)")
    parser.add_argument("--dry-run", action="store_true",
                        help="only print the resolved config and the run id")
    args = parser.parse_args()
    run_experiment(args.config, args.overrides, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
