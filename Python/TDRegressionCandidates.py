#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_cell_magic('capture', '', '%pip install nfl_data_py --upgrade\n')


# In[3]:


import pandas as pd

import nfl_data_py as nfl
import numpy as np
import warnings; warnings.simplefilter('ignore')


# In[4]:


# finding data from 2008 2022
# importing directly from nfl_data_py
seasons = range(2008, 2022)
df = nfl.import_pbp_data(seasons)
df.head()


# In[5]:


df.shape


# In[6]:


# way for us to create for loop to find the relevant columns since the dataset is so large
for column in df.columns:
    if 'rush' in column:
        print(column)
    elif 'distance' in column:
        print(column)
    elif 'yardline' in column:
        print(column)


# In[7]:


# creating a new dataframe made from these 4 columns
rushing_df = df[['rush_attempt', 'rush_touchdown', 'yardline_100', 'two_point_attempt']]


# In[8]:


# sorting this new dataframe to where there was a rush attempt, as the rush_attempt and two_point_attempt columns are yes for 1 or no for 0. I forget what the number indexing is called for 1 or 0
rushing_df = rushing_df.loc[(rushing_df['rush_attempt'] == 1 & (rushing_df['two_point_attempt'] == 0))]

rushing_df.sort_values(by='yardline_100')


# In[9]:


# making a new dataframe grouping yardline and rush touchdown, counting the total number of occurences.
# We then set this to a new dataframe rushing_df_probs.
# When looking at the new dataframe, it shows the proprortion for each yardline.
# For example, looking at the 1 yardline, there are 2100 touchdowns and 1722 not touchdowns on a total of 
# 3822 attempts
rushing_df_probs = rushing_df.groupby("yardline_100")['rush_touchdown'].value_counts(normalize=True)

# turned series into data frame
rushing_df_probs = pd.DataFrame({
    'probability_of_touchdown': rushing_df_probs.values
}, index=rushing_df_probs.index).reset_index()

rushing_df_probs


# In[10]:


rushing_df_probs = rushing_df_probs.loc[rushing_df_probs['rush_touchdown'] == 1]
rushing_df_probs = rushing_df_probs.drop('rush_touchdown', axis=1)
rushing_df_probs.head(15)


# In[11]:


import seaborn as sns; sns.set_style('whitegrid')

rushing_df_probs.plot(x='yardline_100', y='probability_of_touchdown')


# In[12]:


# importing 2022 data as pbp_2022
pbp_2022 = nfl.import_pbp_data([2022])

# sorting 2022 by player, playerid, posteam, rush touchdown, and yardline
pbp_2022 = pbp_2022[['rusher_player_name', 'rusher_player_id' , 'posteam', 'rush_touchdown', 'yardline_100']].dropna()


# In[13]:


pbp_2022.head()


# In[14]:


exp_df = pbp_2022.merge(rushing_df_probs, how='left', on= 'yardline_100')
exp_df.head()


# In[15]:


import numpy as np

# editing the exp_df dataframe by grouping by rushing player name, id, and posteam


exp_df = exp_df.groupby(['rusher_player_name','rusher_player_id', 'posteam'], as_index=False).agg({
    'probability_of_touchdown': np.sum,
    'rush_touchdown': np.sum
# renaming probability and rush_touchdown as seen below 

}).rename({
    'probability_of_touchdown': 'Expected Touchdowns',
    'rush_touchdown': 'Actual Touchdowns'
}, axis=1) #whenever renaming or editing a column must pass axis=1 argument

exp_df = exp_df.sort_values(by='Expected Touchdowns', ascending=False)
exp_df.head()


# In[16]:


# renaming columns
exp_df = exp_df.rename({
    "rusher_player_name": "Player",
    "posteam": "Team",
    "rusher_player_id": "ID"
}, axis=1)


exp_df.head()


# In[17]:


# imported weekly data for roster information
roster = nfl.import_weekly_data([2022])
roster.head()

# renamed two columns to match for the exp_df table so i can merge them down below
roster = roster[['player_id', 'position']].rename({
    'player_id': 'ID'
}, axis=1)


exp_df = exp_df.merge(roster, on='ID', how='left')

exp_df.head()



# In[18]:


exp_df = exp_df.loc[exp_df['position'] == 'RB'].drop('position',axis=1)

# dropped duplicate rows because as you can see above I was getting 5 copies of ja.williams lol
exp_df = exp_df.drop_duplicates()

exp_df.head()


# In[19]:


# Ranked the amount of touchdowns per player
exp_df['Actual Touchdowns Rank'] = exp_df['Actual Touchdowns'].rank(ascending=False)

exp_df


# In[25]:


# new column named regression candidate
exp_df['Regression Candidate'] = exp_df['Expected Touchdowns'] - exp_df['Actual Touchdowns']

# sorting them
exp_df.sort_values(by='Regression Candidate', ascending=True)


# In[81]:


from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(12,8))



exp_df['Positive Regression Candidate'] = exp_df ['Regression Candidate'] > 0

# creating a scatter plot
sns.scatterplot(
    hue='Positive Regression Candidate', 
    x='Expected Touchdowns', 
    y='Actual Touchdowns', 
    data = exp_df,
    palette=['r','g'],
).set(title='Positive Regression Candidates')

ax.legend()


# these variables help draw the line to the last point, which is what causes this diagonal line
max_act_touchdowns = int(exp_df['Actual Touchdowns'].max())
max_exp_touchdowns = int(exp_df['Expected Touchdowns'].max())

max_tds = max(max_act_touchdowns, max_exp_touchdowns)

sns.lineplot(x=range(max_tds),y=range(max_tds))

notable_players = ['T.Etienne', 'Ja.Williams', 'A.Ekeler']

for _, row in exp_df.iterrows():
    if row['Player'] in notable_players:
        ax.text(
            x=row['Expected Touchdowns'] + .1,
            y=row['Actual Touchdowns'] + 0.05,
            s=row['Player'] 
        )


# In[ ]:





# In[ ]:




