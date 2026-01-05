import argparse
from pathlib import Path
from src import config as cfg
from src.runners import ExperimentRunner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--data-path", default=None)
    args = parser.parse_args()

    if args.data_path:
        cfg.RAW_DATA_PATH = Path(args.data_path)

    runner = ExperimentRunner(cfg)
    run_id = runner.train(model_name=args.model, run_id=args.run_id)
    print(f"Run saved: reports/experiments/{run_id}")

if __name__ == "__main__":
    main()
