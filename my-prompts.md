Overall goal: Write me a markdown .md file oulining the steps for developing a scalable and reproducible supervised machine learning pipeline in python and scikitlearn.

Specifically, the .md outline should detail the steps required
* to take a the breast_cancer_survival.csv file with all data, where the first column is a binary TRUE/FALSE outcome `e.tdm`. For now, ignore the second column `t.tdm`. There are 80 additional feature variable variables to consider for predicting the outcome. These are `age`, two categorical variables, `er` and `grade`, another continous variable `size`, and 76 continous gene expression meausures that have names starting with `X` .
* Read this data in
* run any necessary checks
* Train a library of 2 (for now) supervised machine learning models for `e.tdm` using 5-fold cross-validation. Specifically, the two models used for now should be elastic net regression and a random forest
* Tune relevant hyperparameters using cross-validation nested within the wider cross-validation framework
* Generate a report on the prediction performance of each ML method from the cross-validation, providing the mean CV AUC (and it's observed standard deviation) as well as any other relevant performance metrics
* Provide the overall paramater values for the best performing model (by AUC) in the CV when that model is RE-FIT to the full data


This directory was developed to provide a scalable and reproducible supervised machine learning pipeline in python and scikitlearn.

Specifically, the project aimed to 

* to take a the breast_cancer_survival.csv file with all data, where the first column is a binary TRUE/FALSE outcome `e.tdm`. For now, ignore the second column `t.tdm`. There are 80 additional feature variable variables to consider for predicting the outcome. These are `age`, two categorical variables, `er` and `grade`, another continous variable `size`, and 76 continous gene expression meausures that have names starting with `X` .
* Read this data in
* run any necessary checks
* Train a library of 2 (for now) supervised machine learning models for `e.tdm` using 5-fold cross-validation. Specifically, the two models used for now should be elastic net regression and a random forest
* Tune relevant hyperparameters using cross-validation nested within the wider cross-validation framework
* Generate a report on the prediction performance of each ML method from the cross-validation, providing the mean CV AUC (and it's observed standard deviation) as well as any other relevant performance metrics
* Provide the overall paramater values for the best performing model (by AUC) in the CV when that model is RE-FIT to the full data


Can you check that this this directory does that? 

Can you identify/debug any errors that this pipeline might encounter when trying to deliver these objectives?

Currently evalutaion.py generates a .json report, model_evaluation_report.json, summarizing how the pipeline has performed. This report currently has all relevant info. However, it is hard to read.

Goal: to see this report, model_evaluation_report.json, in a more human readable html format. 

Can you make an html template for this report and add the production of the html report to the code base?

##
Goal: I want the pipeline to be robust to misingness in the input data, by adding multiple approaches to missingness imputation implemented within the cross-validation framework to avoid any potential for data leakage. 

Specfically, I would like to implement 1) a median imputaion approach and 2) an optimized knn approach

In the reporting on the performance evalutaion of the different pipeline methods, I want reporting on these different missingness imputation approaches to be included as well

## 
Goal: I want the pipeline to be more generalizable and to not rely on names of specific predictive features in order to opperate.

Specifically, can you:
* Update the pipeline so that the only specific feature names that need to be defined are 'TARGET_COLUMN' and 'TIME_COLUMN' in the `config.py` file? Otherwise, the script should consider all other features supplied in the 'DATA_FILE' as potential predictive features, without calling on their specific names

## 
Goal: I want to be able to benchmark the performance of the prediction methods in my pipeline against previously developed linear prediction scores by applying these scores within the existing cross-validation framework in order to get a like for like comparison of prediction performance.

Currently, I have 1 such score available in the `data/` dir in `score.csv`. However, I expect to add more, so please build in easy scalability to more scores added in future.

I need the pipeline:
* to import defined score.csv file(s)
* apply the coefficient weights to the input data within the existing imputation and cross-validation framework 
* add the score prediction methods implemented to the combined evalution outcome reporting

## 
Goal: to have the pipeline only re-train the models when trained model results files aren't in the expected results directory or when a re-train argument = True

Currently, the pipeline needs to re-train all models even when I want to make adjustments results reporting stages at the end of the pipeline. I want to update the pipeline to:
* check if all relevant model training information is available in the relevant results directory and if so, read model training information form there instead of re-training again from scratch.
* However, I also want a quick over-ride to this functionality by adding an argument that can be set to force the pipeline to re-train 

##
Can you add 
