import argparse
import logging

import wandb
import wandb.apis.reports as wr

from .config import WandBConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

ENTITY, PROJECT = WandBConfig.ENTITY, WandBConfig.PROJECT


def main(wandb_id: str):

    # Extract wandb runs
    api = wandb.Api()
    queried_run = api.run(f"{ENTITY}/{PROJECT}/{wandb_id}")
    base_run = list(api.runs(f"{ENTITY}/{PROJECT}", filters={"tags": "baseline"}))[0]

    logging.info("Comparing the runs.")
    logging.info(f"Queried run: {queried_run.name}")
    logging.info(f"Baseline run: {base_run.name}\n")

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

    logging.info(f"Report created at {report.url}!")

    return report.url


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare the runs.")
    parser.add_argument(
        "--wandb_id",
        help="WandB ID to compare the runs, create a report for the run with this ID compared to the baseline run.",
    )
    args = parser.parse_args()
    main(args.wandb_id)
