# IEEE-CIS Fraud Detection

Imagine standing at the check-out counter at the grocery store with a long line behind you and the cashier not-so-quietly announces that your card has been declined.
In this moment, you probably aren’t thinking about the data science that determined your fate. Embarrassed, and certain you have the funds to cover everything needed 
for an epic nacho party for 50 of your closest friends, you try your card again. Same result. As you step aside and allow the cashier to tend to the next customer,
you receive a text message from your bank. “Press 1 if you really tried to spend $500 on cheddar cheese.”. While perhaps cumbersome (and often embarrassing) in the moment,
this fraud prevention system is actually saving consumers millions of dollars per year.

The task in this project therefore is to classify whether a transaction is fraudulent given a large-scale dataset. The data comes from Vesta's real-world e-commerce transactions 
and contains a wide range of features from device type to product features. If successful, you’ll improve the efficacy of fraudulent transaction alerts for millions of people
around the world, helping hundreds of thousands of businesses reduce their fraud loss and increase their revenue.
And of course, you will save party people just like you the hassle of false positives.

## Installation
### Downloading the Data
- Clone this repository to your computer
- Navigate to the project directory `cd packages/random_forest_model/random_forest_model` from your terminal
- Run `mkdir input`
- Use `cd input` to go into the directory where the data is to be stored
- Download the data files from Kaggle
  - [Click here](https://www.kaggle.com/c/ieee-fraud-detection/data) to go straight to the Data page on Kaggle
  - If you don't have a Kaggle account then you'd have to create one

### Installing the Requirements 
- Navigate to the Random Forest Package using `cd packages/random_forest_model` 
- Install the requirements using `pip install -r requirements`
To install requirements for the API: 
- Navigate to the API using `cd packages/ml_api`
- Install the requirements using `pip install -r requirements`

## Usage
- Navigate to `cd packages/random_forest_model/random_forest_model` from your terminal
  - Run `python train.py`
  - This will train 5 Random Forest Models using 100 estimators on different parts of the training data (I used 5-fold Stratified CV) and save them to a directory called
`models` with a serialized Label Encoder included.

**Disclaimer:** *This project has not yet been deployed so is not yet available to be consumed.*

## Extending This Work
Some ideas to extend this work include: 
- Use an XGBoost algorithm instead of Random Forest
- Connecting to CI/CD Pipeline
   - Attempted this but the action fails due to lack of memory in the Free version 
- Differntial Testing 
- Deploying to a PaaS without a container (Heroku is a good option) 
- Run this app in a Container using Docker
- Deploying to IaaS such as AWS ECS 
- Try Deep Learning

## Write-Ups related to this project
- [Random Forest Overview](https://medium.com/me/stats/post/746e7983316?source=main_stats_page)
- [Structuring Machine Learning Projects](https://medium.com/me/stats/post/be473775a1b6?source=main_stats_page)
- [Comprehension of the AUC-ROC Curve](https://towardsdatascience.com/comprehension-of-the-auc-roc-curve-e876191280f9)
- [Using Machine Learning to Detect Fraud](https://towardsdatascience.com/using-machine-learning-to-detect-fraud-f204910389cf)
- [Serving a Machine Learning Model via REST API](https://towardsdatascience.com/serving-a-machine-learning-model-via-rest-api-5a4b38c02e90)
