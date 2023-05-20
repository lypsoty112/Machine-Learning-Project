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
from sklearn.impute import SimpleImputer, KNNImputer
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

def train(file):
    # Load the csv file
    df = pd.read_csv(file)
    log(f"Loaded {file} successfully", 0)
    # Drop the ID column
    df = df.drop(columns=["ID"])

    # Split the data into training and test sets
    X = df.drop(columns=["ROBOT"])
    y = df["ROBOT"]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
    log(f"Prepared the data successfully, Shape: {X_train.shape}, {y_test.shape}", 0)
    
    # --------------------------------------------------------------------------------------------------------
    best_pipeline = [
        ("imputer", KNNImputer(n_neighbors=10)),
        ("scaler", RobustScaler()),
        ("model", RandomForestClassifier())
    ]

    # Create a grid of hyperparameters
    param_grid = {
        "n_estimators": [1, 100, 200], # n_estimators = number of trees
        "max_depth": [None, 10, 20], # max_depth = max depth of the tree
        "min_samples_split": [1, 10, 20], # min_samples_split = min number of samples required to split an internal node
        "min_samples_leaf": [1, 3, 5], # min_samples_leaf = min number of samples required to be at a leaf node
    }

    limit = 25

    grid_steps = {
        "n_estimators": int((param_grid["n_estimators"][2] - param_grid["n_estimators"][1]) / limit),
        "max_depth": int((param_grid["max_depth"][2] - param_grid["max_depth"][1]) / limit),
        "min_samples_split": int((param_grid["min_samples_split"][2] - param_grid["min_samples_split"][1]) / limit),
        "min_samples_leaf": int((param_grid["min_samples_leaf"][2] - param_grid["min_samples_leaf"][1]) / limit),
    }

    for key in grid_steps:
        if grid_steps[key] < 1:
            grid_steps[key] = 1

    best_params = {
        "n_estimators": None,
        "max_depth": None,
        "min_samples_split": None,
        "min_samples_leaf": None,
    }


    # Create the grid search
    # Perform a big grid search 

    # Process the data 
    X_train_scaled = best_pipeline[1][1].fit_transform(best_pipeline[0][1].fit_transform(X_train))
    X_test_scaled = best_pipeline[1][1].transform(best_pipeline[0][1].transform(X_test))

    for i in range(limit):
        log(f"Grid search {i+1}/{limit}" + " " * 20)
        grid_search = GridSearchCV(best_pipeline[2][1], param_grid, verbose=0, n_jobs=-1, scoring="f1")
        # Fit the grid search
        grid_search.fit(X_train_scaled, y_train)

        # Get the best parameters
        new_params = {
            "n_estimators": grid_search.best_params_["n_estimators"],
            "max_depth": grid_search.best_params_["max_depth"],
            "min_samples_split": grid_search.best_params_["min_samples_split"],
            "min_samples_leaf": grid_search.best_params_["min_samples_leaf"],
        }

        # Compare the new params to the old params
        if all([best_params[key] == new_params[key] for key in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]]):
            # If the parameters haven't changed, break out of the loop
            log("The grid search has converged.", 1)
            break

        best_params = new_params
        # Change the param_grid
        param_grid = {
            "n_estimators": [
                best_params["n_estimators"] - grid_steps["n_estimators"] if best_params["n_estimators"] > 1 else 1, 
                best_params["n_estimators"], 
                best_params["n_estimators"] + grid_steps["n_estimators"]
            ],
            "max_depth": [
                best_params["max_depth"] - grid_steps["max_depth"] if best_params["max_depth"] is not None else None, 
                best_params["max_depth"], 
                best_params["max_depth"] + grid_steps["max_depth"] if best_params["max_depth"] is not None else grid_steps["max_depth"]
            ],
            "min_samples_split": [
                best_params["min_samples_split"] - grid_steps["min_samples_split"] if best_params["min_samples_split"] > 1 else 1, 
                best_params["min_samples_split"], 
                best_params["min_samples_split"] + grid_steps["min_samples_split"]
            ],
            "min_samples_leaf": [
                best_params["min_samples_leaf"] - grid_steps["min_samples_leaf"] if best_params["min_samples_leaf"] > 1 else 1, 
                best_params["min_samples_leaf"], 
                best_params["min_samples_leaf"] + grid_steps["min_samples_leaf"]
            ],
        }

    # Perform 1 final grid search with the best parameters

    param_grid = {
        "n_estimators": [best_params["n_estimators"]], # n_estimators = number of trees
        "max_depth": [best_params["max_depth"]], # max_depth = max depth of the tree
        "min_samples_split": [best_params["min_samples_split"]], # min_samples_split = min number of samples required to split an internal node
        "min_samples_leaf": [best_params["min_samples_leaf"]], # min_samples_leaf = min number of samples required to be at a leaf node
        "max_features": ["auto", "sqrt", "log2"], # max_features = number of features to consider when looking for the best split
        "bootstrap": [True, False], # bootstrap = whether bootstrap samples are used when building trees
        "criterion": ["gini", "entropy"], # criterion = function to measure the quality of a split
        "class_weight": ["balanced", "balanced_subsample", None], # class_weight = weights associated with classes
    }
    # Create the grid search
    grid_search = GridSearchCV(best_pipeline[2][1], param_grid, verbose=0, n_jobs=-1, scoring="f1")
    grid_search.fit(X_train_scaled, y_train)
    best_params = {
        "n_estimators": grid_search.best_params_["n_estimators"],
        "max_depth": grid_search.best_params_["max_depth"],
        "min_samples_split": grid_search.best_params_["min_samples_split"],
        "min_samples_leaf": grid_search.best_params_["min_samples_leaf"],
        "max_features": grid_search.best_params_["max_features"],
        "bootstrap": grid_search.best_params_["bootstrap"],
        "criterion": grid_search.best_params_["criterion"],
        "class_weight": grid_search.best_params_["class_weight"],
    }
    log(f"Best parameters: {grid_search.best_params_}", 0)

    # --------------------------------------------------------------------------------------------------------
    # Train based on the best parameters
    best_pipeline = Pipeline([
        ("imputer", KNNImputer(n_neighbors=10)),
        ("scaler", RobustScaler()),
        ("model", RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"]
        ))
    ])
    best_pipeline.fit(X_train, y_train)

    # Get the training and test accuracy
    train_acc = best_pipeline.score(X_train, y_train)
    test_acc = best_pipeline.score(X_test, y_test)

    log(f"Training Accuracy: {train_acc}")
    log(f"Test Accuracy: {test_acc}")

    # Get the predictions
    from sklearn.metrics import classification_report

    predictions = best_pipeline.predict(X_test)

    # Print the classification report
    log(classification_report(y_test, predictions))


    # Save the model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_pipeline, f)

def check(file):
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

def predict(file):    
    # Load the model
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
    log(f"Pipeline loaded successfully: \n{pipeline}", 0)
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

def main():
    
    # Check if arg 1 exists
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "train"
    file = sys.argv[2].lower() if len(sys.argv) > 2 else "data/weblogs_train.csv"
    # Assert the file exists & is a csv file
    assert file.endswith(".csv"), "File must be a csv file"
    assert os.path.exists(file), "File does not exist"

    if mode == "train":
        train(file=file)
    elif mode == "check":   
        check(file=file)
    else:
        predict(file=file)

if __name__ == "__main__":
    main()