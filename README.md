# Predictive Analytics for Financial Fraud Detection in Digital Transactions
---

## Key Takeaways
This project successfully developed an XGBoost-based fraud detection model for synthetic financial transactions. Key achievements include:
*   **Robust Feature Engineering**: Critical features such as datetime components, log-transformed transaction amounts, balance deltas, and transaction counts for both originators and destinations were engineered. The `is_high_risk_type` feature also proved valuable.
*   **Effective Handling of Imbalanced Data**: The highly imbalanced nature of the fraud dataset was addressed effectively using `scale_pos_weight` within the XGBoost model, ensuring that the model did not simply predict the majority class.
*   **Time-Series Cross-Validation**: The use of `TimeSeriesSplit` prevented data leakage and provided a more realistic evaluation of the model's performance on unseen, future data.
*   **Strong Model Performance**: The XGBoost model achieved excellent performance metrics on the test set, including an ROC-AUC of 0.9992 and a PR-AUC of 0.9848, demonstrating its ability to accurately identify fraudulent transactions with a high recall (0.9212) and precision (0.9909).
*   **Interpretability**: Feature importance analysis highlighted key drivers of fraud, such as destination-related transaction history (`nameDest_avg_amount`, `num_trx_f`) and balance changes (`deltaOrig`), offering valuable insights for understanding fraudulent patterns.
  
## 🎯Overview:
This project aims to develop a robust fraud detection model using a financial trasaction dataset that contains mobile money transactions based on a sample of real transactions extracted from one month of financial logs from a mobile money service. The objective is to accurately identify and flag fraudulent activities within a simulated financial environment, thereby mitigating potential financial losses and enhancing security. The challenge lies in distinguishing subtle patterns of fraudulent behavior from legitimate transactions, often complicated by the dynamic and adaptive nature of fraud schemes.

To achieve this, we are utilizing the [Paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1) dataset, a synthetic dataset that mirrors real-world transaction data, providing a rich environment to explore and model various types of financial fraud. Its comprehensive nature makes it an ideal resource for building and evaluating effective fraud detection systems against financial fraud.

Extracted financial trasaction dataset from Kaggle using Kaggle API:

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
- __Error Analysis__


__#### Data Extraction:__ 
The dataset was acquired by downloading it from Kaggle using the `opendatasets` library. Specifically, the command `od.download('https://www.kaggle.com/datasets/ealaxi/paysim1')` was used to fetch the data, which was then loaded into a pandas DataFrame using `pd.read_csv('./paysim1/PS_20174392719_1491204439457_log.csv')`.

__#### Data Cleaning & Preprocessing:__
Initial data cleaning involved several steps to ensure data quality and prepare for time-series analysis:
- *Checking for Missing Values and Duplicates*: A thorough check was performed using `file.isna().sum()` and `file.duplicated().sum()`. The results confirmed no missing values or duplicate rows were present in the dataset.
- *Sorting by 'step' column*: The dataset was initially sorted by the `step` column using `file=file.sort_values(by='step', ascending=True)`. This was crucial as `step` represents a temporal order, and maintaining this order is essential for subsequent time-series feature engineering and splitting.
- *Converting 'step' to 'datetime'*: The numerical `step` column, which represents an hour in the simulation, was converted into a proper `datetime` column. An imaginary start date of '2010-01-01 00:00:00' was chosen, and `pd.to_timedelta(file['step'], unit='h')` was added to this base date to create the `datetime` column. The dataset was then re-sorted by this new `datetime` column, and the index was reset.
- *Dropping 'isFlaggedFraud' Column*: The `isFlaggedFraud` column was dropped because it was considered redundant for our modeling objective. This column specifically flags transactions that meet a very strict definition of fraud (transfer attempts of over 200,000 units to a single destination account). While related to fraud, our goal is to build a model that identifies a broader range of fraudulent transactions captured by the `isFraud` column, which includes all types of fraudulent activities. Relying solely on `isFlaggedFraud` would limit the model's ability to generalize to other fraud patterns.

__#### Feature Engineering:__
Several new features were engineered to enhance the model's ability to detect fraud:
- *Time-based Features*: From the `datetime` column, `hour`, `day`, and `weekday` features were extracted. These capture cyclical patterns and temporal trends in fraudulent activities.
- *Logarithmic Transformation of Amount*: `log_amount = np.log1p(df['amount'])` was created to handle the skewed distribution of transaction amounts, making the feature more suitable for modeling.
- *Transaction Counts and Average Amounts*: `nameOrig_count` and `nameDest_count` represent the cumulative number of transactions for the originator and destination accounts, respectively. `nameOrig_avg_amount` and `nameDest_avg_amount` calculate the cumulative average transaction amounts for these accounts. These features help identify unusual transaction volumes or amounts for specific entities.
- *Balance Change Indicators*: `deltaOrig` (`oldbalanceOrg - newbalanceOrig - amount`) and `deltaDest` (`newbalanceDest - oldbalanceDest - amount`) capture discrepancies in expected balance changes, which are strong indicators of fraud. Additionally, `orig_balance_zero`, `dest_balance_zero`, and `no_balance_change` were created to flag specific balance conditions that might be associated with fraudulent transactions.
- *Total Transaction Counts*: `num_trx` (total transactions by originator) and `num_trx_f` (total transactions by destination) along with `avg_amnt_f` (average amount for destination) provide further insights into the transaction behavior of entities.
- *High-Risk Transaction Type*: A binary feature `is_high_risk_type` was created to flag `TRANSFER` and `CASH_OUT` transaction types, as these were identified during EDA as having a higher propensity for fraud.

The dataset was split into training and testing sets based on a `cutoff_date` of '2010-01-22'. This time-based split, with `train = file[file['datetime'] < cutoff_date].copy()` and `test = file[file['datetime'] >= cutoff_date].copy()`, was crucial to prevent data leakage and ensure that the model is evaluated on future, unseen data, reflecting a real-world fraud detection scenario.

__#### EDA:__
Summary of EDA Insights:

**1. Transaction Type Distributions and Fraud Rates:**
-   *Transaction Volume:* 'CASH_OUT' and 'PAYMENT' are the most frequent transaction types, accounting for a significant majority of all transactions. 'CASH_IN' also has a high volume, while 'TRANSFER' and 'DEBIT' are less common.
-   *Fraud Rate by Type:* 'TRANSFER' transactions exhibit the highest fraud rate (around 0.55%), followed by 'CASH_OUT' (around 0.13%). 'CASH_IN', 'PAYMENT', and 'DEBIT' transactions show almost no fraudulent activity in the dataset.

**2. Fraud Trends Over Time (Hourly/Daily):**
-   *Hourly Trends:* The fraud rate shows a distinct pattern throughout the day. Fraud activity significantly peaks in the early morning hours, particularly between 2 AM and 7 AM (with the highest spike around 5 AM, reaching ~19% fraud rate), and is much lower during typical business hours and evenings.
-   *Daily Trends:* Fraud rates also vary by day, with Monday (weekday 0) and Tuesday (weekday 1) showing slightly higher fraud rates compared to other days. There's no clear linear trend, but specific days have noticeable increases.

**3. Amount Distribution:**
-   *Overall Distribution:* The histogram of transaction amounts, especially on a logarithmic scale, reveals that a large number of transactions involve smaller amounts, with a long tail extending to very large amounts.
-   *Fraud by Amount Bucket:* The heatmap of fraud rate by transaction type and amount bucket indicates that fraud predominantly occurs in 'TRANSFER' and 'CASH_OUT' types, and it is observed across various amount ranges, though some higher amount buckets within these types might show increased fraud rates.

**4. Data Imbalance:**
-   The dataset exhibits a severe class imbalance. Legitimate transactions ('isFraud' = 0) comprise an overwhelming majority (99.91%), while fraudulent transactions ('isFraud' = 1) are extremely rare (only 0.09%). This imbalance is clearly visible in the `isFraud` value counts and the pie chart.
-   *Implication:* This severe imbalance is a critical challenge for model training. A model trained without addressing this imbalance might achieve high overall accuracy by simply predicting the majority class, but it would perform poorly in detecting the minority (fraudulent) class. Therefore, appropriate handling techniques (such as `scale_pos_weight` used in XGBoost, or other oversampling/undersampling methods) are essential for building an effective fraud detection model.

__#### Model Training & Evaluation:__
We chose XGBoost (Extreme Gradient Boosting) for this fraud detection task due to several key advantages:
1.  **Handling Imbalanced Datasets**: Fraud detection datasets are inherently imbalanced, meaning there are far fewer fraudulent transactions than legitimate ones. XGBoost effectively addresses this challenge through the `scale_pos_weight` parameter, which assigns higher weight to the minority class (fraudulent transactions). This helps the model learn from and correctly classify rare fraud events without being overwhelmed by the majority class.
2.  **Performance and Efficiency**: XGBoost is a highly efficient and powerful gradient boosting framework known for its speed and accuracy. It is well-suited for large datasets and complex relationships between features. The `tree_method='hist'` and `device='gpu'` parameters were utilized to leverage GPU acceleration, significantly speeding up the training process on a large dataset like this one.
3.  **Robustness**: As an ensemble method, XGBoost builds multiple decision trees sequentially, correcting errors of previous trees. This makes it robust to noise and outliers, which are common in real-world transactional data.

__TimeSeriesSplit Cross-Validation Strategy__
For time-series data, traditional k-fold cross-validation is inappropriate because it can lead to data leakage by allowing the model to be trained on future data to predict past data. To prevent this and ensure a realistic evaluation of our model's performance over time, we employed `TimeSeriesSplit` cross-validation.

`TimeSeriesSplit` works by splitting the dataset into training and validation sets where the validation set always comes *after* the training set chronologically. For each fold:
- The training data consists of observations up to a certain point in time.
- The validation data consists of subsequent observations.

In our implementation, we used `n_splits=5`, creating five different train-validation splits. To ensure consistent training data sizes for each fold and to prevent potential issues with very small initial training sets, we used a `fixed_train_size` of 1,000,000 samples for each training fold. This approach mimics a real-world scenario where a model is trained on past data and evaluated on new, unseen data, providing a more reliable estimate of its generalization performance.

__Performance Metrics on Training Folds__
During the TimeSeriesSplit cross-validation, the model demonstrated strong and consistent performance across all five folds. The average scores across all folds are as follows:

-   **Accuracy**: 0.9999
-   **F1 Score**: 0.9242
-   **ROC-AUC**: 0.9987
-   **PR-AUC**: 0.9797
-   **Precision**: 0.9011
-   **Recall**: 0.9505

The high average ROC-AUC score of approximately 0.9987 indicates the model's excellent ability to discriminate between fraudulent and legitimate transactions, suggesting a very low rate of misclassification across various thresholds. The F1 score, precision, and recall also reflect a well-balanced performance in identifying positive (fraudulent) cases.

__#### Pediction and Evaluation:__
After training, the model was evaluated on a completely unseen, held-out test set (`test` dataframe). The performance metrics on this final test set are:

```
               Accuracy       F1   ROC-AUC    PR-AUC  Precision    Recall
XGBClassifier   0.99924  0.95479  0.998827  0.984787   0.990905  0.921214
```

The model maintained its strong performance on the unseen test data. Notably, the ROC-AUC score of **0.9988** reaffirms its exceptional capability in distinguishing fraud. The F1 score of 0.9548, precision of 0.9909, and recall of 0.9212 indicate a robust model that can effectively identify fraudulent transactions with high confidence and minimal false positives, which is critical in fraud detection systems.

__#### Error Analysis:__
The confusion matrix provides a clear breakdown of the model's predictions versus the actual labels on the test set. From the heatmap, we can identify the following:

- **True Negatives (TN):** The model correctly predicted legitimate transactions as legitimate. These are transactions where `isFraud = 0` and the model predicted `0`.
- **False Positives (FP):** The model incorrectly predicted legitimate transactions as fraudulent. These are transactions where `isFraud = 0` but the model predicted `1`.
- **False Negatives (FN):** The model incorrectly predicted fraudulent transactions as legitimate. These are transactions where `isFraud = 1` but the model predicted `0`.
- **True Positives (TP):** The model correctly predicted fraudulent transactions as fraudulent. These are transactions where `isFraud = 1` and the model predicted `1`.

From the confusion matrix, we have the following counts:
- **True Negatives (TN): 295970**
- **False Positives (FP): 22**
- **False Negatives (FN): 205**
- **True Positives (TP): 2397**

Let's analyze the implications of these errors in the context of fraud detection:

**False Positives (FP): 22 transactions**
False positives occur when the model incorrectly flags a legitimate transaction as fraudulent. In a financial system, this means:
- *Customer Inconvenience:* Legitimate transactions might be declined or put on hold, leading to frustrated customers who may need to verify their identity or approve the transaction manually. This can negatively impact user experience and trust.
- *Operational Costs:* The bank or financial institution may incur operational costs associated with investigating these false alerts, contacting customers, and resolving issues.
- *Low Number:* With only 22 false positives out of approximately 296,000 legitimate transactions, the model demonstrates excellent precision. This low number suggests that the model is very good at not bothering legitimate customers unnecessarily.

**False Negatives (FN): 205 transactions**
False negatives are more critical in fraud detection as they represent actual fraudulent transactions that the model failed to identify. This means:
- *Direct Financial Loss:* Each false negative represents a missed fraud event, leading directly to financial losses for the bank or its customers.
- *Reputation Damage:* Repeated missed frauds can lead to reputational damage and a loss of customer trust.
- *Higher Impact:* While the number (205) is significantly higher than false positives, it still represents a small fraction of the total fraudulent transactions that occurred (2397 TP + 205 FN = 2602 actual frauds). The model's recall indicates that it catches a high percentage of actual fraud cases, but these 205 missed frauds are still a concern.

**Balancing Errors:**
In fraud detection, there's often a trade-off between minimizing false positives and minimizing false negatives. A high recall (catching most frauds) might lead to more false positives, while a high precision (few false alarms) might miss more frauds. Our model shows a good balance:
- **Precision: 0.9909** (meaning ~99.09% of transactions flagged as fraud are actually fraud)
- **Recall: 0.9212** (meaning ~92.12% of actual fraud cases are detected)

The high precision is beneficial as it minimizes customer friction and operational overhead, while the high recall ensures that a large majority of fraudulent activities are caught.

__Insights from Precision-Recall (PR) and Receiver Operating Characteristic (ROC) Curves__

**Precision-Recall Curve (PR-AUC): 0.9848**
The Precision-Recall curve is particularly informative for imbalanced datasets like ours, where the positive class (fraud) is rare. A high PR-AUC score of **0.9848** indicates that the model is very effective at retrieving positive instances (fraudulent transactions) while maintaining a high precision. This means that when the model predicts fraud, it's highly likely to be correct, and it doesn't miss many actual fraud cases. This is crucial for operational efficiency, as it minimizes the number of legitimate transactions flagged for manual review.

**Receiver Operating Characteristic (ROC-AUC): 0.9988**
The ROC curve plots the True Positive Rate (Recall) against the False Positive Rate at various threshold settings. An exceptionally high ROC-AUC score of **0.9988** signifies that the model has outstanding discriminative power. It can effectively distinguish between the positive and negative classes across almost all possible classification thresholds. This means that the model is very good at separating fraudulent from legitimate transactions. The curve's steep rise to the top-left corner on the plot confirms that a high recall can be achieved with a very low false positive rate, demonstrating the model's strong overall performance.


## 🛠️ Tools & Technologies
- __Programming Language:__ Python
- __Libraries:__ NumPy, Pandas, Seaborn, Matplotlib, Scikit-learn, XGBoost
- __Environment:__ Google Colab

The project will deliver a robust and scalable classification model capable of predicting customer churn. Further this model can be integrated with an application or deployed as a lone model to predict customer churn.
