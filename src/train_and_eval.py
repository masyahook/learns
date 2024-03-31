import argparse
import logging

import numpy as np
import wandb
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
        project="ci-cd-for-ml-gitops-course",
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

        # Log information about the data and target
        logging.info("Data shape: %s", X.shape)
        logging.info("Target shape: %s", y.shape)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
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

        # Log that the model has been trained
        logging.info("Model trained")

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Log that the model has been tested
        logging.info("Model tested")

        # Calculate the classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_str = classification_report(y_test, y_pred)

        run.log({"classification_report": report})

        # Log the classification report using logging
        logging.info("Classification Report:\n%s", report_str)


if __name__ == "__main__":
    main()
