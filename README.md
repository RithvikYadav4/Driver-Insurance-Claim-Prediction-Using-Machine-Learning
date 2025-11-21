# Driver-Insurance-Claim-Prediction-Using-Machine-Learning

This project focuses on predicting whether a driver will file an insurance claim the next year using various machine learning models and sampling strategies. The dataset is highly imbalanced, making evaluation and sampling crucial for reliable model performance.

# Dataset Overview
1.Dataset Name :Porto Seguro train dataset

2.Total Rows: 595,212

3.Total Features: 59 (anonymized by Porto Seguro)

4.Target Column: target
0 → No insurance claim
1 → Insurance claim

5.Imbalance Ratio: ~96:4

6.Missing Values: None in the provided train data

7.Feature Types:
bin → Binary features
cat → Categorical features
ord → Ordinal-like encoded features
reg → Continuous features
ps_ind_, ps_car_, ps_reg_ → Names anonymized
calc features often considered less informative/noisy
All features are anonymous due to privacy regulations.

# Sampling Techniques Applied
1.Random Undersampling (RUS)

2.Random Oversampling (ROS)

3.Cluster Centroid Sampling

4.SMOTE (Synthetic Minority Oversampling Technique)

Each method was evaluated with multiple ML models to observe performance differences.

# Machine Learning Models Used

1.Logistic Regression

2.Decision Tree Classifier

3.K-Nearest Neighbors (KNN)

# Evaluation metrics included

1.F1-score

2.AUC-ROC

3.Confusion Matrix

# Result
1.Best Undersampling Model

Logistic Regression (RUS)

F1-score: 0.0948

AUC-ROC: 0.6223

Demonstrated robust performance on reduced majority data, balancing recall and false positives effectively.

2.Best Oversampling Model

Logistic Regression (ROS)

F1-score: 0.0951

AUC-ROC: 0.6218

Showed stable generalization after synthetic upsampling, outperforming Decision Tree and KNN.

3.Cluster Centroid Sampling Result

a)Led to model collapse, producing single-class predictions.

b)Indicates this strategy is unsuitable for high-dimensional, heavily imbalanced data.

4.SMOTE Performance

a)Improved class balance

b)But resulted in less stable F1/AUROC compared to simple RUS & ROS

c)Shows that SMOTE may not be optimal for this dataset’s structure

# Technologies Used

1.Python

2.Jupyter Notebook / Google Colab

3.Numpy,Pandas,Matplotlib

4.Scikit-learn 

5.Imbalanced-learn (imblearn)

6.RandomUnderSampler,RandomOverSampler,ClusterCentroids,SMOTE

7.Logistic Regression,Decision Tree Classifier,KNN

8.F1-Score,AUC-ROC,Confusion Matrix

