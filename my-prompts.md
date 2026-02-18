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