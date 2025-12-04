# Predictive Analytics for Financial Fraud Detection in Digital Transactions
---
## 🎯Overview:
This project aims to develop a robust fraud detection model using a financial trasaction dataset that contains mobile money transactions based on a sample of real transactions extracted from one month of financial logs from a mobile money service. The objective is to accurately identify and flag fraudulent activities within a simulated financial environment, thereby mitigating potential financial losses and enhancing security. The challenge lies in distinguishing subtle patterns of fraudulent behavior from legitimate transactions, often complicated by the dynamic and adaptive nature of fraud schemes.

To achieve this, we are utilizing the Paysim1 dataset, a synthetic dataset that mirrors real-world transaction data, providing a rich environment to explore and model various types of financial fraud. Its comprehensive nature makes it an ideal resource for building and evaluating effective fraud detection systems against financial fraud.

Extracted financial trasaction dataset from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) using Kaggle API:

- __Synthetic Financial Datasets For Fraud Detection__


In this project, we have utilized the above-mentioned dataset, which contains contains **6.36 million records** of mobile money transactions extracted from one month of financial logs from a mobile money service with 11 attributes. Key features include:

| Feature              | Feature meaning                                                                                                           |
|----------------------|---------------------------------------------------------------------------------------------------------------------------|
| **step**             | maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation)        |
| **type**             | type of transaction (CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER)                                                      |
| **amount**           | amount of the transaction in local currency                                                                               |
| **nameOrig**         | customer who started the transaction                                                                                      |
| **oldbalanceOrg**    | initial balance before the transaction                                                                                    |
| **newbalanceOrig**   | new balance after the transaction                                                                                         |
| **nameDest**         | customer who is the recipient of the transaction                                                                          |
| **oldbalanceDest**   | initial balance recipient before the transaction                                                                          | 
| **newbalanceDest**   | new balance recipient after the transaction                                                                               |
| **isFraud**          | This is the transactions made by the fraudulent agents                                                                    |
| **isFlaggedFraud**   | The business model aims to control massive transfers from one account to another and flags illegal attempts               |


## ⚙️Methods
The project employs time series analysis techniques, including:

- __Data Extraction__
- __Data Cleaning & Preprocessing__
- __Feature Engineering__
- __EDA__
- __Model Training & Evaluation__
- __Pediction and Evaluation__

  

## 🛠️ Tools & Technologies
- __Programming Language:__ Python
- __Libraries:__ NumPy, Pandas, Seaborn, Matplotlib, Scikit-learn, XGBoost
- __Environment:__ Google Colab

The project will deliver a robust and scalable classification model capable of predicting customer churn. Further this model can be integrated with an application or deployed as a lone model to predict customer churn.
