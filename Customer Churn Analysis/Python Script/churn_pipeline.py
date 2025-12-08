#!/usr/bin/env python
# coding: utf-8

# # Customer Churn Analysis
# Notebook: data generation → EDA → feature engineering → modeling → evaluation → save outputs
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib, os


# In[18]:


os.makedirs('output_churn', exist_ok=True)


# In[21]:


import random
import numpy as np


# In[22]:


np.random.seed(42)
n = 3000


# In[29]:


np.random.seed(48)
print('random.seed')


# # --- Data generation (columns shown) --

# In[2]:


pd.read_csv(r'C:\Users\hp\Downloads\churn_cleaned.csv')


# In[33]:


df.head()


# In[34]:


df.info()


# In[38]:


df.describe()


# In[39]:


df['churn'].value_counts(normalize=True)


# In[43]:


df.to_csv('output_churn/churn_cleaned.csv', index=False)


# # EDA plots
#  #churn distribution

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import roc_curve, auc
# load cleaned data (adjust path if needed)
df = pd.read_csv(r'C:\Users\hp\Downloads\churn_cleaned.csv')   
df.head()


# In[4]:


ct = pd.crosstab(df['contract'], df['churn'])
ct.columns = ['No','Yes']   # 0 -> No, 1 -> Yes

ax = ct.plot(kind='bar', figsize=(8,5))
ax.set_title('Count of Churn by Contract Type')
ax.set_xlabel('Contract Type')
ax.set_ylabel('Count of Customers')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# # Churn proportion by Contract type

# In[5]:


ct_pct = ct.div(ct.sum(axis=1), axis=0)   # row-wise percentage
ax = ct_pct.plot(kind='bar', stacked=True, figsize=(8,5))
ax.set_title('Proportion of Churn by Contract Type (Stacked)')
ax.set_xlabel('Contract Type')
ax.set_ylabel('Proportion')
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))
plt.xticks(rotation=0)
plt.legend(title='Churn')
plt.tight_layout()
plt.show()


# # Churn rate by Internet Service and Payment Method

# In[6]:


pivot = pd.crosstab(index=df['internet_service'], columns=df['payment_method'], values=df['churn'], aggfunc='mean').fillna(0)
# pivot contains churn rate (0-1) for each cell
fig, ax = plt.subplots(figsize=(10,5))
pivot.plot(kind='bar', ax=ax)
ax.set_title('Churn Rate by Internet Service and Payment Method')
ax.set_xlabel('Internet Service')
ax.set_ylabel('Churn Rate')
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))
plt.xticks(rotation=0)
plt.legend(title='Payment Method', bbox_to_anchor=(1.02,1), loc='upper left')
plt.tight_layout()
plt.show()


# # Boxplot: Monthly charges by churn

# In[7]:


fig, ax = plt.subplots(figsize=(7,5))
data_no = df[df.churn==0]['monthly_charges']
data_yes = df[df.churn==1]['monthly_charges']
ax.boxplot([data_no, data_yes], labels=['No Churn','Churn'])
ax.set_title('Monthly Charges Distribution by Churn')
ax.set_ylabel('Monthly Charges')
plt.tight_layout()
plt.show()


# # Violin-like: density overlay using hist

# In[8]:


plt.figure(figsize=(8,5))
plt.hist(df[df.churn==0]['monthly_charges'], bins=30, alpha=0.6, label='No Churn')
plt.hist(df[df.churn==1]['monthly_charges'], bins=30, alpha=0.6, label='Churn')
plt.title('Monthly Charges Distribution (Churn vs No Churn)')
plt.xlabel('Monthly Charges')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.show()


# # churn rate by tenure bucket

# In[9]:


group = df.groupby('tenure_bucket')['churn'].mean().reindex(['0-3','4-6','7-12','13-24','25-60','60+'])
ax = group.plot(kind='bar', figsize=(8,4))
ax.set_title('Churn Rate by Tenure Bucket')
ax.set_xlabel('Tenure Bucket (months)')
ax.set_ylabel('Churn Rate')
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# # Correlation matrix for numeric fields

# In[10]:


num_cols = ['tenure','monthly_charges','support_calls','num_products','support_per_month','churn']
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(7,6))
cax = ax.imshow(corr, vmin=-1, vmax=1)
ax.set_xticks(range(len(num_cols))); ax.set_yticks(range(len(num_cols)))
ax.set_xticklabels(num_cols, rotation=45, ha='right'); ax.set_yticklabels(num_cols)
# annotate values
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha='center', va='center', fontsize=9)
fig.colorbar(cax, fraction=0.046, pad=0.04)
ax.set_title('Correlation Matrix (numeric features)')
plt.tight_layout()
plt.show()


# # Support calls vs monthly charges colored by churn

# In[11]:


plt.figure(figsize=(8,6))
mask_no = df['churn']==0
mask_yes = df['churn']==1
plt.scatter(df[mask_no]['support_calls'], df[mask_no]['monthly_charges'], alpha=0.5, label='No')
plt.scatter(df[mask_yes]['support_calls'], df[mask_yes]['monthly_charges'], alpha=0.6, label='Yes')
plt.title('Support Calls vs Monthly Charges (colored by churn)')
plt.xlabel('Support Calls')
plt.ylabel('Monthly Charges')
plt.legend(title='Churn')
plt.tight_layout()
plt.show()


# # Joining of two Tables

# In[17]:


import pandas as pd


# In[18]:


df_clean = pd.read_csv(r'C:\Users\hp\Downloads\churn_cleaned.csv')          
df_preds = pd.read_csv(r'C:\Users\hp\Downloads\predictions_for_powerbi.csv')


# In[19]:


print(df_clean.columns)
print(df_preds.columns)


# In[22]:


# Reset index so both start at 0–N
df_clean = df_clean.reset_index(drop=True)
df_preds = df_preds.reset_index(drop=True)

# Concatenate side-by-side
df_joined = pd.concat([df_clean, df_preds], axis=1)


# In[23]:


df_joined.head()


# In[24]:


df_joined.to_csv("output_churn/churn_final_joined_by_index.csv", index=False)


# In[25]:


df_joined.head()


# # Top 10 high-risk customers

# In[31]:


preds = pd.read_csv(r'C:\Users\hp\Desktop\output_churn\churn_final_joined_by_index.csv')   
topN = preds.sort_values('pred_proba_rf', ascending=False).head(10)
display_cols = ['customerID','monthly_charges','pred_proba_rf','pred_rf','revenue_loss_estimate']
print("Top 10 predicted high-risk customers (Random Forest):")
topN[display_cols].reset_index(drop=True)


# # ROC curve for model

# In[32]:


# Cell 11: ROC curve
from sklearn.metrics import roc_curve, auc
preds = pd.read_csv('output_churn/predictions_for_powerbi.csv')
fpr, tpr, _ = roc_curve(preds['actual_churn'], preds['pred_proba_rf'])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'--')   # diagonal
plt.title(f'ROC Curve (RF) — AUC = {roc_auc:.3f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.tight_layout()
plt.show()


# # Cohort-ish retention view (tenure progression)

# In[33]:


churn_by_tenure = df.groupby('tenure')['churn'].mean().reset_index()
plt.figure(figsize=(10,4))
plt.plot(churn_by_tenure['tenure'], churn_by_tenure['churn'], marker='o')
plt.title('Churn Rate by Exact Tenure (months)')
plt.xlabel('Tenure (months)')
plt.ylabel('Churn Rate')
plt.ylim(0, churn_by_tenure['churn'].max()*1.1)
plt.grid(True, linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.show()


# In[3]:


customer_id = [f"CUST{100000+i}" for i in range(n)]


# In[4]:


tenure = np.random.exponential(scale=12, size=n).astype(int)
monthly_charges = np.round(np.random.normal(loc=70, scale=30, size=n).clip(5, 300), 2)
total_charges = np.round((tenure * monthly_charges) + np.random.normal(0, 50, n), 2)
contract_type = np.random.choice(["Month-to-month","One year","Two year"], p=[0.55,0.25,0.20], size=n)
payment_method = np.random.choice(["Electronic check","Mailed check","Bank transfer","Credit card"], size=n)
internet_service = np.random.choice(["DSL","Fiber optic","No"], p=[0.35,0.45,0.20], size=n)
support_calls = np.random.poisson(lam=1.2, size=n)


# In[39]:


df.head()


# In[5]:


num_products = np.random.choice([1,2,3], p=[0.6,0.3,0.1], size=n)
autopay = np.random.choice([0,1], p=[0.4,0.6], size=n)


# In[6]:


paperless_billing = np.random.choice([0,1], p=[0.4,0.6], size=n)
senior_citizen = np.random.choice([0,1], p=[0.88,0.12], size=n)
gender = np.random.choice(["Male","Female"], size=n)
partner = np.random.choice([0,1], size=n)
dependents = np.random.choice([0,1], size=n)


# # churn probability (logistic model)

# In[7]:


logit = (-1.5 + 0.03*(monthly_charges-70) - 0.02*tenure +
         0.6*(contract_type=="Month-to-month").astype(int) +
         0.4*(internet_service=="Fiber optic").astype(int) +
         0.5*(support_calls>2).astype(int) -0.7*autopay + 0.25*paperless_billing)
prob = 1 / (1 + np.exp(-logit))
churn = np.random.binomial(1, prob, size=n)


# In[8]:


df = pd.DataFrame({
    'customerID': customer_id, 'tenure':tenure, 'monthly_charges':monthly_charges,
    'total_charges':total_charges, 'contract':contract_type, 'payment_method':payment_method,
    'internet_service':internet_service, 'support_calls':support_calls, 'num_products':num_products,
    'autopay':autopay, 'paperless_billing':paperless_billing, 'senior_citizen':senior_citizen,
    'gender':gender, 'partner':partner, 'dependents':dependents, 'churn':churn
})


# In[40]:


df.info()


# # features

# In[35]:


df['tenure_bucket'] = pd.cut(df.tenure, bins=[-1,3,6,12,24,60,200], labels=['0-3','4-6','7-12','13-24','25-60','60+'])
df['high_monthly_flag'] = (df.monthly_charges > df.monthly_charges.quantile(0.75)).astype(int)
df['support_per_month'] = df.apply(lambda r: r.support_calls / max(1, r.tenure), axis=1)


# In[36]:


df.head()


# In[37]:


os.makedirs('output_churn', exist_ok=True)
df.to_csv('output_churn/churn_cleaned.csv', index=False)


# In[38]:


df.head()


# In[ ]:





#  # Modelling

# In[12]:


X = df.drop(columns=['customerID','churn','total_charges'])
y = df.churn
numeric_features = ['tenure','monthly_charges','support_calls','num_products','support_per_month']
categorical_features = ['contract','payment_method','internet_service','autopay','paperless_billing','senior_citizen','gender','partner','dependents','tenure_bucket','high_monthly_flag']

num_pipe = Pipeline([('scaler', StandardScaler())])
cat_pipe = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore'))])
preproc = ColumnTransformer([('num', num_pipe, numeric_features), ('cat', cat_pipe, categorical_features)])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

pipe_lr = Pipeline([('preproc', preproc), ('clf', LogisticRegression(max_iter=1000))])
pipe_rf = Pipeline([('preproc', preproc), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])

pipe_lr.fit(X_train, y_train)
pipe_rf.fit(X_train, y_train)


# # Evaluate

# In[13]:


for name, pipe in [('LR', pipe_lr), ('RF', pipe_rf)]:
    preds = pipe.predict(X_test); probs = pipe.predict_proba(X_test)[:,1]
    print(name, 'acc', accuracy_score(y_test,preds), 'roc_auc', roc_auc_score(y_test,probs),
          'precision', precision_score(y_test,preds), 'recall', recall_score(y_test,preds))


# # predictions CSV for Power BI

# In[14]:


out = X_test.copy().reset_index(drop=True)
out['actual_churn'] = y_test.values


# In[42]:


df.head()


# In[15]:


out['pred_proba_rf'] = pipe_rf.predict_proba(X_test)[:,1]
out['pred_proba_lr'] = pipe_lr.predict_proba(X_test)[:,1]


# In[16]:


out['pred_rf'] = pipe_rf.predict(X_test)
out['pred_lr'] = pipe_lr.predict(X_test)
out['revenue_loss_estimate'] = out['pred_proba_rf'] * out['monthly_charges'] * 3  


# In[17]:


out.to_csv('output_churn/predictions_for_powerbi.csv', index=False)
joblib.dump(pipe_rf, 'output_churn/random_forest_pipeline.joblib')
joblib.dump(pipe_lr, 'output_churn/logistic_pipeline.joblib')


# In[45]:


df.describe()


# In[47]:


out = X_test.copy().reset_index(drop=True)
out['actual_churn'] = y_test.values
out['pred_proba_rf'] = pipe_rf.predict_proba(X_test)[:,1]
out['pred_proba_lr'] = pipe_lr.predict_proba(X_test)[:,1]
out['pred_rf'] = pipe_rf.predict(X_test)
out['pred_lr'] = pipe_lr.predict(X_test)
out['revenue_loss_estimate'] = out['pred_proba_rf'] * out['monthly_charges'] * 3
out.to_csv('output_churn/predictions_for_powerbi.csv', index=False)


# In[46]:


df.info()


# In[49]:


from sklearn.metrics import roc_curve, auc
import pandas as pd

y_test = [0, 1, 0, 1, 1, 0, 1, 0]  
y_pred_proba = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4]  

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

print("AUC Score:", roc_auc)



# In[52]:


# Save AUC value for Power BI
pd.DataFrame({"AUC":[roc_auc]}).to_csv("output_churn/auc_value.csv", index=False)



# In[53]:


# Save ROC curve points for Power BI visual (optional)
pd.DataFrame({"FPR":fpr, "TPR":tpr}).to_csv("output_churn/roc_curve_points.csv", index=False)


# In[1]:


get_ipython().run_line_magic('run', 'scripts/model_training.py')


# In[3]:


get_ipython().run_line_magic('run', 'correct_path/model_training.py')

# Option 2: Create the directory structure and file if it doesn't exist
# First, create the directory (if needed)
import os
if not os.path.exists('scripts'):
    os.makedirs('scripts')
    
# Then create the file with your code
get_ipython().run_line_magic('%writefile', 'scripts/model_training.py')
# Your model training code here
# For example:
import numpy as np
from sklearn.linear_model import LinearRegression

# Define your model training code
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y)
print(f"Model score: {reg.score(X, y)}")


# In[5]:


import sys
sys.path.append("scripts")

import model_training
model_training.train_model()
model_training.evaluate_model()



# In[ ]:


# Option 1: Convert the notebook to a Python module first
# (Run this in a separate cell or terminal)
# !jupyter nbconvert --to python path/to/your_notebook.ipynb

# Then import it as a regular module
import sys
sys.path.append("path/to")  # Add the directory containing the converted .py file
import your_notebook  # This imports the converted Python file

# Call functions from the notebook
your_notebook.train_model()
your_notebook.evaluate_model()

# Option 2: Use nbimport to directly import the notebook
# First install nbimport if you don't have it
# !pip install nbimport

import nbimport
import sys
sys.path.append("path/to")  # Add the directory containing the .ipynb file
import your_notebook  # This imports the .ipynb file directly

# Call functions from the notebook
your_notebook.train_model()
your_notebook.evaluate_model()


# In[ ]:


get_ipython().system('jupyter nbconvert --to python')

