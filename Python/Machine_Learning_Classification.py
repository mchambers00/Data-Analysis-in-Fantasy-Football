#!/usr/bin/env python
# coding: utf-8

# In[12]:


get_ipython().run_cell_magic('capture', '', '%pip install nfl-data-py --quiet\n')


# In[13]:


import nfl_data_py as nfl
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from matplotlib import pyplot as plt

import warnings; warnings.simplefilter('ignore')


# In[14]:


weekly_df = nfl.import_weekly_data(years=range(2000, 2022))


# In[15]:


weekly_df.head()


# In[35]:


# making three
eligible_positions = ['WR', 'RB', 'QB']
grouping_columns = ['player_id', 'season']
features = ['targets', 'receptions', 'rushing_yards', 'receiving_yards', 'passing_yards']
target = ['position']

train_df = weekly_df.loc[weekly_df['position'].isin(eligible_positions), grouping_columns + features + target]


# grouping by season instead of week
groupby_funcs = {
    'position': 'first'
}

for feature in features:
    groupby_funcs[feature] = np.sum

train_df = train_df.groupby(grouping_columns, as_index=False).agg(groupby_funcs)

train_df['position'] = train_df['position'].replace({
    'RB':0,
    'WR':1,
    'QB':2
})

# setting parameters for minimum stats required for position

train_df = train_df.loc[(train_df['rushing_yards'] > 200) | (train_df['passing_yards'] > 300) |(train_df['receiving_yards'] > 150)]                                               
train_df.head()


# In[24]:


train_df['position'].unique()


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(
    train_df[features],
    train_df[target],
    test_size=0.2,
    random_state=123
)


# In[44]:


# Creating a decision tree classifier with a max depth of 2
clf = DecisionTreeClassifier(max_depth=2)


# Fitting the data using x_train and y_train
clf.fit(X_train, y_train)

# making a variable to predict against the y_test
y_pred = clf.predict(X_test)

y_pred

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


# In[47]:


fig, ax = plt.subplots(figsize=(10,6))

class_names = ['RB', 'WR', 'QB']


# Left is true right is false for the first variable rushing_yards <= 116.5
plot_tree(clf, ax=ax, feature_names=features, class_names=class_names);


# In[55]:


params = {
    'max_depth': range(1,10),
    'min_samples_split': range(2,6),
        }
clf = DecisionTreeClassifier()

# applies 10 fold cross validation, splits data into 90% training, 10% test
grid_search = GridSearchCV(clf, params, cv=10)

grid_search.fit(train_df[features],train_df[target])


# In[64]:


best_clf = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print ('Best Parameters:', best_params)


# In[60]:


best_clf


# In[63]:


print('Best Score:', best_score)


# In[66]:


# RB = 0, WR = 1,  QB = 2 
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.head()


# In[67]:





# In[ ]:




