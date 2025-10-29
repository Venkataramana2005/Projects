# Customer Churn Prediction

## Objective
This project aims to predict customer churn using machine learning classification techniques. By analyzing historical customer data and behavioral patterns, the model identifies customers at risk of leaving the service, enabling proactive retention strategies and improving customer lifetime value.

## Methodology
The project employs machine learning classification algorithms to predict customer churn. Key steps include:

- **Data Preprocessing**: Cleaning and transforming raw customer data, handling missing values, and encoding categorical variables
- **Feature Engineering**: Selecting and creating relevant features that influence customer churn behavior
- **Model Development**: Training and evaluating multiple classification algorithms to identify the optimal predictive model
- **Performance Evaluation**: Assessing model accuracy using metrics such as precision, recall, F1-score, and ROC-AUC
- **Prediction**: Generating churn probability scores for customers to facilitate targeted retention campaigns

## Requirements
Install the required dependencies using:

```bash
pip install -r requirments.txt
```

The project requires Python 3.x and common data science libraries including pandas, numpy, scikit-learn, and matplotlib.

## Usage Instructions
1. **Data Preparation**: Ensure the dataset files (`WA_Fn-UseC_-Telco-Customer-Churn.csv` and `Test-Customer-Churn.xlsx`) are in the project directory
2. **Run the Notebook**: Open and execute `customerpurchaseproect.ipynb` in Jupyter Notebook or JupyterLab to train the model and generate predictions
3. **Model Training**: Follow the notebook cells sequentially to preprocess data, train the classification model, and evaluate performance
4. **Prediction**: Use the trained model to predict churn probability for new customer data
5. **Analysis**: Review the output metrics and visualizations to understand churn patterns and model performance
