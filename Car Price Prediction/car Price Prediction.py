#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,r2_score
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_csv("car F and P.csv")
data.columns = data.columns.str.strip()
print("\nFirst Some Rows of dataset:")
print(data.head())


# In[3]:


print("Dataset Info:")
print(data.info())


# In[4]:


print("Missing Values:")
print(data.isnull().sum())


# In[5]:


data['Engine Fuel Type'].fillna(data['Engine Fuel Type'].mode()[0],inplace=True)
data['Engine HP'].fillna(data['Engine HP'].median(),inplace=True)
data['Engine Cylinders'].fillna(data['Engine Cylinders'].mode()[0],inplace=True)
data['Market Category'].fillna(data['Market Category'].mode()[0],inplace=True)
data['Number of Doors'].fillna(data['Number of Doors'].mode()[0],inplace=True)


# In[6]:


data.isnull().sum()


# In[7]:


numerical_columns = data.select_dtypes(include=['int64','float64']).columns
numerical_columns = numerical_columns.drop('MSRP')
for col in numerical_columns:
    print(f"\nStatistics of {col}:")
    print(data[col].describe())


# In[8]:


categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"\nStatistics of {col}")
    print(data[col].describe())


# In[9]:


print("Duplicated Rows:")
print(data.duplicated().sum())


# In[10]:


data = data.drop_duplicates()


# In[11]:


print("Duplicated Rows:")
print(data.duplicated().sum())


# In[12]:


sns.histplot(data['MSRP'],bins=50,kde=True)
plt.title("Distribution of MSRP")
plt.xlabel("MSRP")
plt.ylabel("Count")
plt.show()


# In[13]:


sns.scatterplot(x='Engine HP',y='MSRP',data=data)
plt.title("Engine HP Vs MSRP")
plt.xlabel("Engine HP")
plt.ylabel("MSRP")
plt.show()


# In[14]:


sns.boxplot(x='Vehicle Size',y='MSRP',data=data)
plt.title("MSRP by Vehicle Size")
plt.xlabel("Vehicle Size")
plt.ylabel("MSRP")
plt.show()


# In[15]:


plt.figure(figsize=(17,8))
sns.heatmap(data.corr(numeric_only=True),annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# In[16]:


sns.countplot(x='Transmission Type',data=data)
plt.title("Count of Transmission Types")
plt.xticks(rotation=45)
plt.show()


# In[17]:


sns.countplot(x='Number of Doors',data=data)
plt.title("Number of Doors Distribution")
plt.show()


# In[18]:


plt.figure(figsize=(10, 5))
sns.countplot(y='Vehicle Style', data=data, order=data['Vehicle Style'].value_counts().index)
plt.title("Vehicle Style Distribution")
plt.show()


# In[19]:


sns.countplot(x='Driven_Wheels', data=data)
plt.title("Driven Wheels Distribution")
plt.show()


# In[20]:


sns.pairplot(data[['Engine HP', 'Engine Cylinders', 'highway MPG', 'city mpg','MSRP']],hue='MSRP',palette='coolwarm')
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()


# In[21]:


top_makes = data['Make'].value_counts().nlargest(10).index
sns.boxplot(x='Make', y='MSRP', data=data[data['Make'].isin(top_makes)])
plt.title("MSRP by Make (Top 10)")
plt.xticks(rotation=45)
plt.show()


# In[22]:


sns.violinplot(x='Engine Cylinders', y='MSRP', data=data)
plt.title("MSRP by Engine Cylinders")
plt.show()


# In[23]:


avg_msrp_year = data.groupby('Year')['MSRP'].mean().reset_index()
sns.lineplot(x='Year', y='MSRP', data=avg_msrp_year)
plt.title("Average MSRP by Year")
plt.show()


# In[24]:


plt.figure(figsize=(12, 6))
sns.barplot(x='Transmission Type', y='MSRP', hue='Driven_Wheels', data=data)
plt.title("Average MSRP by Transmission Type and Driven Wheels")
plt.xticks(rotation=45)
plt.show()


# In[25]:


numeric_data = data.select_dtypes(include=["float64", "int64"]).drop(columns=["MSRP"])
scaled_data = StandardScaler().fit_transform(numeric_data)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
data['PCA1'] = pca_result[:, 0]
data['PCA2'] = pca_result[:, 1]
sns.scatterplot(x='PCA1', y='PCA2', hue='Vehicle Size', data=data)
plt.title("PCA of Vehicle Specs")
plt.show()


# In[26]:


X = data[['Make', 'Model', 'Year', 'Engine Fuel Type', 'Engine HP',
       'Engine Cylinders', 'Transmission Type', 'Driven_Wheels',
       'Number of Doors', 'Market Category', 'Vehicle Size', 'Vehicle Style',
       'highway MPG', 'city mpg', 'Popularity']]
y = data['MSRP']


# In[27]:


preprocessor = ColumnTransformer(
    transformers = [
        ('numerical',StandardScaler(),numerical_columns),
        ('categorical',OneHotEncoder(handle_unknown='ignore'),categorical_columns)
    ]
)


# In[28]:


lr_model = Pipeline([
    ("preprocessor",preprocessor),
    ("model",LinearRegression())
])


# In[29]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[30]:


lr_model.fit(X_train,y_train)


# In[31]:


y_pred = lr_model.predict(X_test)
print("---------------- Linear Regression Results ---------------")
print(f"\nMean Squared Error (MSE): {mean_squared_error(y_test,y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test,y_pred):.2f}")


# In[32]:


rf_model = Pipeline([
    ("preprocessor",preprocessor),
    ("model",RandomForestRegressor(random_state=42))
])


# In[33]:


rf_model.fit(X_train,y_train)


# In[34]:


y_pred = rf_model.predict(X_test)
print("--------------- Random Forest Results ---------------")
print(f"\nMean Squared Error (MSE): {mean_squared_error(y_test,y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test,y_pred):.2f}")


# In[35]:


param_grid_rf = {
    "model__n_estimators": [50,100],
    "model__max_depth": [None,10,20],
    "model__min_samples_split": [2,5]
}


# In[36]:


grid_search = GridSearchCV(rf_model,param_grid_rf,cv=3,scoring="r2",n_jobs=-1)
grid_search.fit(X_train,y_train)

best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)


# In[37]:


print("--------------- Tuned Random Forest Results ---------------")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test,y_pred):.2f} ")
print(f"R2 Score: {r2_score(y_test,y_pred):.2f}")


# In[38]:


gb_model = Pipeline([
    ("preprocessor",preprocessor),
    ("model",GradientBoostingRegressor(random_state=42))
])


# In[39]:


gb_model.fit(X_train,y_train)


# In[40]:


y_pred = gb_model.predict(X_test)
print("--------------- Gradient Boosting Regressor Results ---------------")
print(f"Mean Sqaured Error (MSE): {mean_squared_error(y_test,y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test,y_pred):.2f}")


# In[41]:


xgb_model = Pipeline([
    ("preprocessor",preprocessor),
    ("model",XGBRegressor(random_state=42))
])


# In[42]:


xgb_model.fit(X_train,y_train)


# In[43]:


y_pred = xgb_model.predict(X_test)
print("--------------- XGB Regressor Results ---------------")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test,y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test,y_pred):.2f}")


# In[44]:


data['predicted MSRP'] = np.maximum(0,lr_model.predict(X))
data['predicted MSRP'] = data['predicted MSRP'].replace(0, np.nan)


# In[45]:


plt.scatter(y, data['predicted MSRP'], alpha=0.6) 

min_val = min(y.min(), data['predicted MSRP'].min())
max_val = max(y.max(), data['predicted MSRP'].max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')

plt.title('Predicted MSRP vs. Actual MSRP', fontsize=16)
plt.xlabel('Actual MSRP', fontsize=14)
plt.ylabel('Predicted MSRP', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend() 
plt.tight_layout() 
plt.show()


# In[48]:


data.drop(columns=['PCA1','PCA2'],inplace=True)


# In[49]:


data.to_csv("Predicted_Car_Price.csv",index=False)
print("Predicted MSRP added to the dataset and saved.")

