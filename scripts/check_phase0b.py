from pathlib import Path

from paper2.common.config import load_yaml
from paper2.eval.metrics_spec import print_metric_summary


def main():
    exp_cfg = load_yaml(Path("configs/experiment.yaml"))

    print("Loaded experiment:", exp_cfg["experiment"]["name"])
    print("Current stage:", exp_cfg["experiment"]["current_stage"])

    print("\n=== Modes ===")
    for k, v in exp_cfg["modes"].items():
        print(f"{k}: enabled={v['enabled']}")

    print("\n=== Output dirs ===")
    for k, v in exp_cfg["outputs"].items():
        print(f"{k}: {v}")

    print("\n=== Freeze policy ===")
    print("Frozen now:")
    for item in exp_cfg["freeze_policy"]["frozen_now"]:
        print(" -", item)

    print("\nNot frozen yet:")
    for item in exp_cfg["freeze_policy"]["not_frozen_yet"]:
        print(" -", item)

    print("\n=== Metric Summary ===")
    print_metric_summary()

    phase0_doc = Path("docs/phase0_freeze.md")
    print("\nPhase0 document exists:", phase0_doc.exists())


if __name__ == "__main__":
    main()