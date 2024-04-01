# `masyahook` learns

This repo contains my personal learnings around various topics in Machine Learning, CI/CD, and other tech stuff. I will be updating this repo as I learn new things.

## Things I learnt

### [CI/CD for Machine Learning](https://www.wandb.courses/courses/ci-cd-for-machine-learning)

As part of CI/CD for Machine Learning course by Weights and Biases, we implemented a few modules and workflows. The corresponding Weights and Biases project can be found [here](https://wandb.ai/masyahook/ci-cd-for-ml-gitops-course). Please checkout [`hw/wandb-run-base-comparison`](https://github.com/masyahook/learns/tree/hw/wandb-run-base-comparison) branch.

- `src/train_and_eval.py` - a dummy pipeline that trains and evaluates a machine learning model. Everything is logged in Weights and Biases (you can find the project [here](https://wandb.ai/masyahook/ci-cd-for-ml-gitops-course)). To run the pipeline, you can use the following command:

  ```bash
  python -m src.train_and_eval --model_type <model_type> --tags <tags>
  ```

- `src/compare_runs.py` - a script that compares the runs of a project in Weights and Biases. You can pass `wandb_run_id` to compare corresponding run with `baseline` run. To run the script, you can use the following command:

  ```bash
  python -m src.compare_runs <wandb_run_id>
  ```

- `.github/workflows/chatops.yaml` - a GitHub Actions workflow that triggers a chatbot to notify the user about the status of the pipeline. To jobs can be triggered:
  - `add-label` - adds a label to the PR using the comment `/label <label>`.
  - `compare-runs` - compares the runs of a project in Weights and Biases using the comment `/wandb <wandb_run_id>`.
