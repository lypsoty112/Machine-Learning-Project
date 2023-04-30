# Machine Learning Project

## FullProject.py
The script `fullProject.py` contains a script for a Random Forest Classifier model that can be used for training, checking, and predicting the classification of a robot. The file can be run in the command line interface by providing arguments such as the mode of operation (train, check or predict), and the file path of the dataset.

## Required libraries
- sys
- os
- pandas
- sklearn
- pickle
- datetime
- matplotlib

`pip install pandas sklearn pickle matplotlib`
## Functions

### `log(message: str, importance: int = 0)`
This function is used for logging messages. It takes two parameters:
- `message` (required): a string containing the message to be logged.
- `importance` (optional): an integer indicating the importance of the message, which can be 0 (INFO), 1 (WARNING), or 2 (ERROR). The default value is 0.

### `main()`
This is the main function of the script, which contains the logic for training, checking, and predicting the Random Forest Classifier model.

## Running the script

To run the script, open a command-line interface, navigate to the directory where the file is located, and enter the following command:

```
python fullProject.py [mode] [file_path]
```

- `fullProject.py`: the name of the Python file.
- `mode`: the mode of operation, which can be "train", "check", or "predict". The default value is "train".
- `file_path`: the path to the dataset file. The default value is "data/weblogs_train.csv".

### Training the model

To train the model, run the following command:
```
python full_project.py train file_path
```

### Checking the model

To check the model, run the following command:
```
python full_project.py check file_path
```

### Predicting with the model

To use the model for prediction, run the following command:
```
python full_project.py predict file_path
```

Note that in this mode, the model should be already trained and saved in a .pkl file. If the file does not exist, the script will raise an error.

## Other files
### splitData.py
This script is used for splitting the dataset file `data/weblogs.csv` into training and testing sets

### Training1.ipynb, Training2.ipynb, Training3.ipynb
These notebooks contain tests & experiments for training the model.