import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

MODEL_PATH = "model/model.pkl"

def log(message: str, importance: int = 0):
    
    date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    importances = ["INFO", "WARNING", "ERROR"]

    importance = importances[importance if 0 <= importance < len(importances) else 0]

    # Print the message
    print(f"[{date}] | [{importance}] {message}")

def main():
    
    # Check if arg 1 exists
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "train"
    file = sys.argv[2].lower() if len(sys.argv) > 2 else "data/weblogs_train.csv"
    # Assert the file exists & is a csv file
    assert file.endswith(".csv"), "File must be a csv file"
    assert os.path.exists(file), "File does not exist"

    if mode == "train":
        # Load the csv file
        df = pd.read_csv(file)
        log(f"Loaded {file} successfully", 0)
        # Prepare the data
        # Drop the columns that are not required
        df.drop('ID', axis=1, inplace=True)
        # Scale the data
        X = df.drop(columns=["ROBOT"])
        y = df["ROBOT"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

        log(f"Prepared the data successfully, Shape: {X_train.shape}, {y_test.shape}", 0)
        best_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", RandomForestClassifier())
        ])

        # Create a grid of hyperparameters
        param_grid = {
            "model__n_estimators": [10, 50, 100, 200],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        }

        # Create the grid search
        grid_search = GridSearchCV(best_pipeline, param_grid, cv=5, verbose=2, n_jobs=-1)
        # Fit the grid search
        grid_search.fit(X_train, y_train)

        log(f"Best parameters: {grid_search.best_params_}", 0)
        # Train based on the best parameters
        best_pipeline = grid_search.best_estimator_
        best_pipeline.fit(X_train, y_train)

        # Get the training and test accuracy
        train_acc = best_pipeline.score(X_train, y_train)
        test_acc = best_pipeline.score(X_test, y_test)

        print(f"Training Accuracy: {train_acc}")
        print(f"Test Accuracy: {test_acc}")

        # Get the predictions
        y_pred = best_pipeline.predict(X_test)

        from sklearn.metrics import classification_report, confusion_matrix

        predictions = best_pipeline.predict(X_test)

        # Print the classification report
        print(classification_report(y_test, predictions))
        print(confusion_matrix(y_test, predictions))

        # Save the model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(best_pipeline, f)
    elif mode == "check":   
        # Check the results
        results = pd.read_csv(file)
        log(f"Loaded {file} successfully", 0)
        # Check the results
        try:
            assert "Predicted" in results.columns, "Predicted column not found"
            assert "ROBOT" in results.columns, "ROBOT column not found"
        except AssertionError as e:
            log(str(e), 2)
            exit(1)
        
        # Get the accuracy
        accuracy = (results["Predicted"] == results["ROBOT"]).sum() / len(results)
        log(f"Accuracy: {accuracy}", 0)

        # Get the confusion matrix
        plt.rc('font', size=10)  # extra code
        ConfusionMatrixDisplay.from_predictions(results["ROBOT"], results["Predicted"],
                                        normalize="true", values_format=".6%", cmap="PuBu")
        plt.show()


    else:
        
        # Load the model
        with open(MODEL_PATH, "rb") as f:
            pipeline = pickle.load(f)

        log(f"Pipeline loaded successfully", 0)
        # Load the csv file
        dfOriginal = pd.read_csv(file)
        # Drop the columns that are not required
        log(f"Loaded {file} successfully", 0)
        # Prepare the data
        # Drop the columns that are not required
        df = dfOriginal.drop('ID', axis=1)
        # Drop the robot column if it exists
        if 'ROBOT' in df.columns:
            df.drop('ROBOT', axis=1, inplace=True)
        X = df
        log(f"Prepared the data successfully, Shape: {X.shape}", 0)
        # Perform the prediction
        y_pred = pipeline.predict(X)
        # Predict the probability
        y_pred_proba = pipeline.predict_proba(X)
        log(f"Prediction performed successfully", 0)

        # Add the prediction to the original dataframe
        dfOriginal['Predicted'] = y_pred
        dfOriginal['Probability_Robot'] = y_pred_proba[:, 1]
        dfOriginal['Probability_Human'] = y_pred_proba[:, 0]

        # Save the dataframe to a csv file
        dfOriginal.to_csv("data/result.csv", index=False)
        log(f"Result saved successfully at data/result.csv", 0)

if __name__ == "__main__":
    main()