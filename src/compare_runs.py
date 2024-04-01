import argparse

import wandb
import wandb.apis.reports as wr

from .config import WandBConfig

ENTITY, PROJECT = WandBConfig.ENTITY, WandBConfig.PROJECT


def main():

    parser = argparse.ArgumentParser(description="Compare the runs.")
    parser.add_argument(
        "--wandb_id",
        help="WandB ID to compare the runs, create a report for the run with this ID compared to the baseline run.",
    )
    args = parser.parse_args()

    # Extract wandb runs
    api = wandb.Api()
    queried_run = api.run(f"{ENTITY}/{PROJECT}/{args.wandb_id}")
    base_run = list(api.runs(f"{ENTITY}/{PROJECT}", filters={"tags": "baseline"}))[0]

    # Extract the classification report from the queried run
    queried_report = queried_run.summary["classification_report"]
    base_report = base_run.summary["classification_report"]

    # Create a report comparing the two runs
    report = wr.Report(
        project=PROJECT,
        title=f"Comparing the runs {queried_run.name} (challenger) and {base_run.name} (baseline)",
        description="Can we compare things automaticallly?",
    )

    report.blocks = [
        wr.TableOfContents(),
        wr.H1("Comparison table"),
        wr.P("Here is a comparison of the classification reports from the two runs."),
        wr.PanelGrid(
            runsets=[
                wr.Runset(
                    ENTITY,
                    PROJECT,
                    "Comparing runs",
                    filters={
                        "display_name": {"$in": [queried_run.name, base_run.name]}
                    },
                ),
            ],
            panels=[
                wr.RunComparer(diff_only="split", layout={"w": 24, "h": 9}),
            ],
        ),
    ]
    report.save()


if __name__ == "__main__":
    main()
