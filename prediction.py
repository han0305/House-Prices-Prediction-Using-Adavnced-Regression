

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
 
from sklearn.preprocessing import StandardScaler
 

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
 
 

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

 
from sklearn.metrics import mean_squared_error
 
pd.pandas.set_option('display.max_columns', None)
 
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('train.csv')
print(data.shape)
print(data.head())

submission = pd.read_csv('test.csv')
print(submission.head())

print(data.dtypes)

print('Number of House Id labels: ', len(data.Id.unique()))
print('Number of Houses in the Dataset: ', len(data))

categorical = [var for var in data.columns if data[var].dtype=='O']
print('There are {} categorical variables'.format(len(categorical)))

numerical = [var for var in data.columns if data[var].dtype!='O']
print('There are {} numerical variables'.format(len(numerical)))

discrete = []
for var in numerical:
    if len(data[var].unique())<20:
        print(var, ' values: ', data[var].unique())
        discrete.append(var)
        
print('There are {} discrete variables'.format(len(discrete)))

for var in data.columns:
    if data[var].isnull().sum()>0:
        print(var, data[var].isnull().mean())

for var in data.columns:
    if data[var].isnull().mean()>0.80:
        print(var, data[var].unique())
        
continuous = [var for var in numerical if var not in discrete and var not in ['Id', 'SalePrice']]
print(continuous)

for var in continuous:
    
    
    plt.subplot(1, 2, 2)
    fig = data[var].hist(bins=20)
    fig.set_ylabel('Number of passengers')
    fig.set_xlabel(var)
 
    plt.show()
for var in discrete:
    print(data[var].value_counts() / np.float(len(data)))
    print()

for var in categorical:
    print(var, ' contains ', len(data[var].unique()), ' labels')
    
 
X_train, X_test, y_train, y_test = train_test_split(data, data.SalePrice, test_size=0.2,random_state=0)
print(X_train.shape, X_test.shape)

for col in continuous:
    if X_train[col].isnull().mean()>0:
        print(col, X_train[col].isnull().mean())
for df in [X_train, X_test, submission]:
    for var in ['LotFrontage', 'GarageYrBlt']:
        df[var+'_NA'] = np.where(df[var].isnull(), 1, 0)
        df[var].fillna(X_train[var].median(), inplace=True) 
 
for df in [X_train, X_test, submission]:
    df.MasVnrArea.fillna(X_train.MasVnrArea.median(), inplace=True)

for col in discrete:
    if X_train[col].isnull().mean()>0:
        print(col, X_train[col].isnull().mean())

for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, X_train[col].isnull().mean())

for df in [X_train, X_test, submission]:
    for var in categorical:
        df[var].fillna('Missing', inplace=True)
submission_vars = []
for var in X_train.columns:
    if var!='SalePrice' and submission[var].isnull().sum()>0:
        print(var, submission[var].isnull().sum())
        submission_vars.append(var)
for var in submission_vars:
    submission[var].fillna(X_train[var].median(), inplace=True)

def tree_binariser(var):
    score_ls = [] 
 
    for tree_depth in [1,2,3,4]:
        
        tree_model = DecisionTreeRegressor(max_depth=tree_depth)
 
        
        scores = cross_val_score(tree_model, X_train[var].to_frame(), y_train, cv=3, scoring='neg_mean_squared_error')
        score_ls.append(np.mean(scores))
 

    depth = [1,2,3,4][np.argmax(score_ls)]

    tree_model = DecisionTreeRegressor(max_depth=depth)
    tree_model.fit(X_train[var].to_frame(), X_train.SalePrice)
    X_train[var] = tree_model.predict(X_train[var].to_frame())
    X_test[var] = tree_model.predict(X_test[var].to_frame())
    submission[var] =  tree_model.predict(submission[var].to_frame())
        
for var in continuous:
    tree_binariser(var)

print(X_train[continuous].head())

def rare_imputation(variable):
    temp = X_train.groupby([variable])[variable].count()/np.float(len(X_train))
    frequent_cat = [x for x in temp.loc[temp>0.03].index.values]
    
    X_train[variable] = np.where(X_train[variable].isin(frequent_cat), X_train[variable], 'Rare')
    X_test[variable] = np.where(X_test[variable].isin(frequent_cat), X_test[variable], 'Rare')
    submission[variable] = np.where(submission[variable].isin(frequent_cat), submission[variable], 'Rare')    
for var in ['BsmtFullBath', 'BsmtHalfBath', 'GarageCars']:
    submission[var] = submission[var].astype('int')
for var in categorical:
    rare_imputation(var)
    
for var in discrete:
    rare_imputation(var)
for var in X_train.columns:
    if var!='SalePrice' and submission[var].isnull().sum()>0:
        print(var, submission[var].isnull().sum())
        submission_vars.append(var)
for var in categorical:
    print(var, X_train[var].value_counts()/np.float(len(X_train)))
    print()
for var in X_train.columns:
    if var!='SalePrice' and submission[var].isnull().sum()>0:
        print(var, submission[var].isnull().sum())

def encode_categorical_variables(var, target):
        
        ordered_labels = X_train.groupby([var])[target].mean().to_dict()
        
        
        X_train[var] = X_train[var].map(ordered_labels)
        X_test[var] = X_test[var].map(ordered_labels)
        submission[var] = submission[var].map(ordered_labels)
 

for var in categorical:
    encode_categorical_variables(var, 'SalePrice')
    
for var in discrete:
    encode_categorical_variables(var, 'SalePrice')
for var in X_train.columns:
    if var!='SalePrice' and submission[var].isnull().sum()>0:
        print(var, submission[var].isnull().sum())
print(X_train.describe())
training_vars = [var for var in X_train.columns if var not in ['Id', 'SalePrice']]

scaler = StandardScaler() 
scaler.fit(X_train[training_vars])
xgb_model = xgb.XGBRegressor()
 
eval_set = [(X_test[training_vars], y_test)]
xgb_model.fit(X_train[training_vars], y_train, eval_set=eval_set, verbose=False)
 
pred = xgb_model.predict(X_train[training_vars])
print('xgb train mse: {}'.format(mean_squared_error(y_train, pred)))
pred = xgb_model.predict(X_test[training_vars])
print('xgb test mse: {}'.format(mean_squared_error(y_test, pred)))

rf_model = RandomForestRegressor()
rf_model.fit(X_train[training_vars], y_train)
 
pred = rf_model.predict(X_train[training_vars])
print('rf train mse: {}'.format(mean_squared_error(y_train, pred)))
pred = rf_model.predict(X_test[training_vars])
print('rf test mse: {}'.format(mean_squared_error(y_test, pred)))

SVR_model = SVR()
SVR_model.fit(scaler.transform(X_train[training_vars]), y_train)
 
pred = SVR_model.predict(scaler.transform(X_train[training_vars]))
print('SVR train mse: {}'.format(mean_squared_error(y_train, pred)))
pred = SVR_model.predict(scaler.transform(X_test[training_vars]))
print('SVR test mse: {}'.format(mean_squared_error(y_test, pred)))

lin_model = Lasso(random_state=2909)
lin_model.fit(scaler.transform(X_train[training_vars]), y_train)
 
pred = lin_model.predict(scaler.transform(X_train[training_vars]))
print('linear train mse: {}'.format(mean_squared_error(y_train, pred)))
pred = lin_model.predict(scaler.transform(X_test[training_vars]))
print('linear test mse: {}'.format(mean_squared_error(y_test, pred)))

pred_ls = []
for model in [xgb_model,rf_model]:
    pred_ls.append(pd.Series(model.predict(submission[training_vars])))
 
pred = SVR_model.predict(scaler.transform(submission[training_vars]))
pred_ls.append(pd.Series(pred))
 
pred = lin_model.predict(scaler.transform(submission[training_vars]))
pred_ls.append(pd.Series(pred))
 
final_pred = pd.concat(pred_ls, axis=1).mean(axis=1)

temp = pd.concat([submission.Id, final_pred], axis=1)
temp.columns = ['Id', 'SalePrice']
temp.head()

temp.to_csv('submit_housesale.csv', index=False)





























