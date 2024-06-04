#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

get_ipython().system('pip install seaborn --upgrade')

df = pd.read_csv('https://raw.githubusercontent.com/fantasydatapros/LearnPythonWithFantasyFootball/master/2023/08-Correlation%20Matrices/weekly_df.csv')
print(df.shape)
df.head()


# In[3]:


df['Position'].unique()


# In[4]:


skill_positions = ['QB', 'TE', 'RB', 'WR']

#isin allows us to filter out row values who's given column value
#is within a list of values.

df = df.loc[df['Position'].isin(skill_positions)]

df.tail()


# In[5]:


df.shape


# In[6]:


import numpy as np

columns = ['Player', 'Tm', 'Position', 'Week', 'PPRFantasyPoints']

new_df = df[columns]
new_df.head()


# In[7]:


new_df = new_df.groupby(['Player', 'Tm', 'Position'], as_index=False).agg({
    'PPRFantasyPoints': np.mean
    
})

new_df.head()


# In[8]:


new_df.loc[(new_df['Position'] == 'WR') & (new_df['Tm'] == 'ARI')].sort_values(by= 'PPRFantasyPoints', ascending=False)


# In[9]:


position_map = {
    'QB': 1,
    'RB':2,
    'WR':3,
    'TE':2
}

def get_top_n_player_at_each_position(df, pos, n):
    df = df.loc[df['Position'] == pos]
    
    return df.groupby('Tm', as_index=True).apply(
        lambda x: x.nlargest(n, ['PPRFantasyPoints']).min()
    )

# get_top_n_player_at_each_position(new_df, 'WR', n=1)

corr_df = pd.DataFrame(columns=columns)

for pos, n_spots in position_map.items():
    for n in range(1, n_spots + 1):
        pos_df = get_top_n_player_at_each_position(new_df, pos, n)
        pos_df = pos_df.rename({'PPRFantasyPoints': f'{pos}{n}'}, axis=1)
        corr_df = pd.concat([corr_df, pos_df], axis=1)
corr_df = corr_df.dropna(axis=1)
corr_df = corr_df.drop(['Position', 'Player', 'Tm'], axis=1)
corr_df.head(32)


# In[10]:


type(corr_df)


# In[11]:


import seaborn as sns
import matplotlib

from matplotlib import pyplot as plt

sns.set_style('whitegrid')
plt.figure(figsize=(12,9))
sns.heatmap(
    corr_df.corr(), annot=True, cmap=sns.diverging_palette(0,250, as_cmap=True), fmt=".2f",
    annot_kws={"size": 10},
    linecolor='gray'
)
plt.title('Heatmap of Correlation Matrix')
plt.show()


# In[ ]:




