## Predicting Unplanned Hospital Readmissions \\ Using Natural Language Processing

### Benjamin Fulton, Dinesh Kasti, Sahibjit Ranu, Jun Zhao

#### How to run the predictor:
1. Navigate to the "code" folder and use the command "sbt run". Four pretrained models have been included in our code submission, corresponding to the four models listed in Figure 3 of our report. The logistic regression is currently set to run. To run a different learner, simply use the command: sbt "run [model]", where [model] is "lr" (logistic regression), "gbt" (gradient boosted trees), "mlp" (multilayer perceptron), or "rf" (random forest). The resulting metrics will be printed to the console, and the predictions saved to "data/test_predictions.csv".
2. The scala code should have automatically generated charts in the "graphics" folder. If for some reason that failed, navigate to the "analysis" folder and run "./make_charts.sh". The resulting charts will be saved to the "graphics" folder. An "environment.yml" file has been included in the "analysis/chart_maker" directory.

#### How to run the preprocessor:
1. Directory and file location set up under folder "CSE6250-NLP", create a folder called "data", put the following three MIMIC III csv files under this folder: NOTEEVENTS.csv, ADMISSIONS.csv, PATIENTS.dsv.
2. Run "python preprocessing/notes_preprocessing.py", and wait until the cleaned up csv file is generated.

#### Generated folder and files:

##### Under "data" folder

* combined.parquet -- parquet file which compiled information from NOTEEVENTS.csv and ADMISSIONS.csv
* feature_importance.csv 
* feature_map.parquet  -- feature map in parquest format
* feature_map.csv -- feature map in csv format
* train.parquet -- training dataset in parquet format
* test.parquet  -- test dataset in parquet format
* predictions_train.csv -- predication results for training dataset
* predictions_validate.csv  -- prediction results for validating dataset
* predictions_test.csv  -- prediction results for testing dataset

##### Under "models" folder

* A folder is generated named based on the learner name and balancing method. For example: RegressionLearnerWithCV.
* Under the newly generated folder, there are four folders: bestModel, estimator, evaluator, metadata.
* Under bestModel and estimator folder, two new folders are generated: metadata, stages.
