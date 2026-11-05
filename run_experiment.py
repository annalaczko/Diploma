"""Run one experiment from a config file (see configs/default.yaml for every
setting and pipeline/experiment.py for what a run records):

    python run_experiment.py configs/vae.yaml
    python run_experiment.py configs/vae.yaml --set clustering.cluster_num=30
    python run_experiment.py configs/vae.yaml --dry-run

A config with a `sweep` runs one experiment per combination of its values
(see pipeline.experiment.run_sweep).
"""

import argparse

from pipeline.experiment import load_config, run_experiment, run_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a pipeline experiment from a YAML config")
    parser.add_argument("config", nargs="?", default=None,
                        help="experiment config, merged over configs/default.yaml (omit to run the defaults)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                        help="override a setting, e.g. --set model.params.k=32 (repeatable)")
    parser.add_argument("--dry-run", action="store_true",
                        help="only print the resolved config and the run id (sweeps: the runs)")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    run = run_sweep if config["sweep"] else run_experiment
    run(args.config, dry_run=args.dry_run, config=config)


if __name__ == "__main__":
    main()
