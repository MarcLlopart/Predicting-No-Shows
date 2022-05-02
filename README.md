# PREDICTING NO SHOW UP IN MEDICAL APPINMENTS

- Sergi Cantón Simó
- Bernat Espinet Torrescassana
- Marc Llopart Enajas

### Objective

We want to predict the show ups in order to optimize the visits and reduce the economic losses.

### EDAs and Preprocessing

At first we checked the data we had to have a little bit of insight of how to manage it. We started by separating some features, creating new ones and then imputate missing data by using a KNN method that worked slightly better than just using the median. Then we checked some correaltions and move forward to delete outliers.

### DATA SPLITTING AND RESAMPLING

As data was clearly unbalanced we needed to balance it to test the model as for example just predicting that everyone will come would give an 80% of accuracy, even that wasn't the metric that was going to be evaluated. We resampled the data by using a RandomOverSampler but not giving a 50/50 data as we thought we would be creating some bias. However, we oversampled the minority class (no show) as an 80% of the majority class.

We also tried a PCA to reduce the model cost. However, it didn't worked as we expected so we decided to move forward without it.
### MODEL SELECTION

We tested XGBoost, Random Forest and Ada Boost in order to the prior assumption of ensemble methods work better. We tried them on a basic form and then tried to improve them by using some hyperparameter tunning. Nevertheless, that seemed to overfit on training data so we kept our previous model. 

We tested SVM but they were extremly slow and didn't improve our previous metrics.

### DATA DASHBOARD
Once we get a working model we decided to show data in an interactive data dashboard. We were interested in doing the dasboard because the buisness instinct says that a visual plot is better than 100 words, and was also new knowledge to us. We decided to plot data distributions for each variable and then we added a model selection so you could see the output file of each model and its statistics as ROC and PR curves and their feature importance to give some buisness insights.



