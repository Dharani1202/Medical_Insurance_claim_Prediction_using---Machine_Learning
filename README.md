# Medical Insurance Claim Prediction Using Machine Learning

This project predicts **medical insurance claim costs** using multiple regression-based machine learning algorithms. The goal is to analyze health and demographic data to estimate individual insurance expenses accurately.

* <a href="https://github.com/Dharani1202/Medical_Insurance_claim_Prediction_using---Machine_Learning/blob/main/ML_Medical%20Insurance_Project.ipynb"> View the Project </a>

## About

This project focuses on predicting the medical insurance charges of individuals based on parameters such as age, gender, BMI, number of children, region, and smoking status.
The process includes **data preprocessing, feature encoding, visualization, model training, and comparison of multiple regression algorithms** to identify the most accurate prediction model.

All analysis is performed using **Python** and documented in a **Jupyter Notebook** for easy understanding.

## Tools & Technologies Used

* **Python (3.x)**
* **Pandas** – for data handling and preprocessing
* **NumPy** – for mathematical operations
* **Matplotlib / Seaborn** – for visualization
* **scikit-learn (sklearn)** – for model building and evaluation
* **SMOTE (from imbalanced-learn)** – for balancing the dataset
* **Jupyter Notebook** – for interactive coding and analysis

## Data Cleaning & Preprocessing

1. **Handle Missing Values and Duplicates**

   * Checked and removed null values using `isnull().sum()`
   * Eliminated duplicate rows for clean and accurate data

2. **Dataset Overview**

   * Explored dataset shape, data types (`dtypes`), column names, and unique values using `nunique()` and `info()`
   * Conducted statistical analysis with `describe()`

3. **Outlier Detection and Removal**

   * Used **Z-Score** and **Box Plots** to identify and remove extreme outliers from numerical columns

4. **Scaling & Bias Reduction**

   * Applied **StandardScaler** to normalize input variables and remove bias in numerical data

5. **Feature Engineering**

   * Separated numerical and categorical columns for specific preprocessing steps
   * Handled **skewness** in data to ensure better model performance

6. **Label Encoding**

   * Used sklearn’s **LabelEncoder** to convert categorical columns like *gender*, *smoker*, and *region* into numerical form

7. **Balancing Dataset**

   * Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance class distribution if required

8. **Train-Test Split**

   * Split the dataset into training and testing subsets using `train_test_split`

## Exploratory Data Analysis (EDA)

### 1. Count of Smokers

Displayed smoker distribution using a **Pie Chart** to visualize the ratio of smokers to non-smokers.

### 2. Distribution of Age

Used **Distplot** to analyze the distribution of age among individuals.

### 3. Gender, Region, and Children Count

Visualized using **Count Plots** to observe how gender, region, and the number of children affect insurance claims.

### 4. Correlation Analysis

Checked correlations among variables using a **heatmap** to understand the relationship between independent and dependent features.

## Machine Learning Model Development

### Algorithms Used

Trained and compared multiple regression models to find the most accurate predictor of insurance claim cost:

* **Linear Regression**
* **Lasso Regression**
* **Decision Tree Regressor**
* **Random Forest Regressor**
* **Gradient Boosting Regressor**
* **AdaBoost Regressor**
* **K-Nearest Neighbors (KNN) Regressor**
* **Support Vector Regressor (SVR)**
* **Bayesian Ridge Regressor**
* **Extra Trees Regressor**

### Model Optimization

* Performed **Grid Search CV** to optimize hyperparameters for better model accuracy and performance.
* Compared results using metrics such as **R² Score**, **Mean Absolute Error (MAE)**, and **Root Mean Squared Error (RMSE)**.

## Evaluation Metrics

Used regression performance metrics for model comparison:

* R² Score
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

## Key Insights

* Age, BMI, and smoking status are strong predictors of medical insurance cost.
* Smokers tend to have significantly higher insurance charges.
* After removing outliers and scaling data, regression models achieved better accuracy.
* Ensemble models like **Random Forest** and **Gradient Boosting** gave higher prediction performance compared to basic linear models.

## Conclusion

This project demonstrates an **end-to-end regression workflow** for predicting medical insurance claims, covering:

* Data cleaning, preprocessing, and visualization
* Feature encoding, scaling, and balancing
* Model training using multiple regression algorithms
* Model evaluation and optimization

The project showcases practical knowledge of **data preprocessing**, **exploratory analysis**, and **machine learning model comparison** in a real-world regression problem.


* <a href="https://github.com/Dharani1202/Medical_Insurance_claim_Prediction_using---Machine_Learning/blob/main/ML_Medical%20Insurance_Project.ipynb"> View the Project </a>

---


