import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
import wandb
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from .config import WandBConfig

ENTITY, PROJECT = WandBConfig.ENTITY, WandBConfig.PROJECT

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():

    parser = argparse.ArgumentParser(
        description="Train and evaluate a machine learning model."
    )
    parser.add_argument(
        "--model_type",
        choices=["logreg", "randomforest", "decisiontree"],
        help="Type of model to train and evaluate",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        help="Tags to associate with the run in Weights and Biases",
    )
    args = parser.parse_args()
    model_type = args.model_type

    # Log the classification report to Weights and Biases
    with wandb.init(
        entity=ENTITY,
        project=PROJECT,
        tags=args.tags,
        config=args,
        job_type=model_type,
    ) as run:

        # Load example data from scikit-learn datasets
        data = load_iris()
        X = data.data
        y = data.target
        # Add noise to X
        noise = np.random.normal(0, 0.5, X.shape)
        X += noise

        # Log the input data and target
        wandb.log(
            {
                "input_X": wandb.Table(
                    dataframe=pd.DataFrame(data.data, columns=data.feature_names)
                ),
                "input_y": wandb.Table(
                    dataframe=pd.DataFrame(data.target, columns=["target"])
                ),
                "noisy_X": wandb.Table(data=X, columns=data.feature_names),
            }
        )

        # Log information about the data and target
        logging.info("Data shape: %s", X.shape)
        logging.info("Target shape: %s", y.shape)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Log the train and test data
        wandb.log(
            {
                "train_X": wandb.Table(data=X_train, columns=data.feature_names),
                "train_y": wandb.Table(
                    dataframe=pd.DataFrame(y_train, columns=["target"])
                ),
                "test_X": wandb.Table(data=X_test, columns=data.feature_names),
                "test_y": wandb.Table(
                    dataframe=pd.DataFrame(y_test, columns=["target"])
                ),
            }
        )

        # Log the size of the train and test sets
        logging.info("Train set size: %s", X_train.shape[0])
        logging.info("Test set size: %s", X_test.shape[0])

        # Train a machine learning model
        match model_type:
            case "logreg":
                model = LogisticRegression()
            case "randomforest":
                model = RandomForestClassifier()
            case "decisiontree":
                model = DecisionTreeClassifier()
            case _:
                logging.error("Invalid model type")
                return

        model.fit(X_train, y_train)

        # Save the trained model using pickle
        model_path = f"{home}/models/trained_{model_type}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        wandb.log_model(name=model_type, path=model_path)

        # Log that the model has been trained
        logging.info("Model trained")

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Log that the model has been tested
        logging.info("Model tested")

        # Calculate the classification report
        report = pd.DataFrame(
            classification_report(y_test, y_pred, output_dict=True)
        ).transpose()
        report_str = classification_report(y_test, y_pred)

        run.log({"classification_report": wandb.Table(dataframe=report)})

        # Log the classification report using logging
        logging.info("Classification Report:\n%s", report_str)


if __name__ == "__main__":
    main()
