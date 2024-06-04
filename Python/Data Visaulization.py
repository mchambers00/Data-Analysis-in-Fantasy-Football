#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import warnings; warnings.simplefilter('ignore')
pd.set_option('display.max_columns', None)


# In[55]:


df = pd.read_csv("https://raw.githubusercontent.com/fantasydatapros/LearnPythonWithFantasyFootball/master/2023/07-Data%20Visualizations/03-Yearly%20Fantasy%20Stats%20-%202022.csv").iloc[:, 1:]


# In[56]:


df.head()


# In[57]:


df.info()


# In[58]:


rb_df = df.loc[df["Pos"] == 'RB']
rb_df['Usage'] = rb_df['Tgt'] + rb_df['RushingAtt']
rb_df['Usage/G'] = rb_df['Usage'] / rb_df['G']
rb_df['FantasyPoints/G'] = rb_df['FantasyPoints'] / rb_df['G'] 
rb_df.head(10)


# In[59]:


plt.figure(figsize=(8,8))
sns.set_style("whitegrid");
sns.scatterplot(x=rb_df['RushingAtt'], y=rb_df['Tgt'],)


# In[60]:


plt.figure(figsize=(8,8))
plt.scatter(rb_df['RushingAtt'], rb_df['Tgt'])
plt.ylabel('Tgt')
plt.xlabel('RushingAtt')
plt.title('Rushing Attempts vs Tgt - 2022 Season', fontsize=16)


# In[61]:


plt.figure(figsize=(12,10))
sns.regplot(x=rb_df['Usage'], y=rb_df['FantasyPoints/G']);


# In[62]:


plt.figure(figsize=(8,8))
sns.kdeplot(rb_df['RushingAtt'])


# In[63]:


plt.figure(figsize=(8,8))
sns.kdeplot(rb_df['Tgt'])


# In[64]:


plt.figure(figsize=(8,8))
sns.displot(rb_df['RushingAtt'], kind='ecdf')


# In[65]:


plt.figure(figsize=(8,8))
sns.displot(rb_df['Tgt'], bins=30)


# In[66]:


fig, ax = plt.subplots(figsize=(10,8))
notable_players = [
    'Austin Ekeler' , 'Aaron Jones' , 'Jamaal Williams'
]
rb_df_filtered = rb_df.loc[rb_df['RushingAtt'] > 50]

for player_name in notable_players:
    player = rb_df_filtered.loc[rb_df_filtered['Player'] == player_name]
    if not player.empty:
        target = player['Tgt']
        rushes = player['RushingAtt']
    ax.annotate(player_name, xy=(rushes+2, target+2), color='red', fontsize=12)
    ax.scatter(rushes, target, color='red')
sns.kdeplot(x=rb_df_filtered['RushingAtt'], y=rb_df_filtered['Tgt'], ax=ax, bw_method=0.7);


# In[67]:


sns.jointplot(x=rb_df_filtered['RushingAtt'], y=rb_df_filtered['Tgt'], kind='hex', dropna=True)


# In[68]:


sns.set_style('dark')

sns.residplot(x=rb_df['Usage/G'], y=rb_df['FantasyPoints/G'])
plt.title('Residual Plot')


# In[69]:


rb_df_copy=rb_df[['RushingAtt', 'RushingTD', 'FantasyPoints/G', 'Tgt']]

sns.pairplot(rb_df_copy, kind='reg')


# In[70]:


weekly_df = pd.read_csv('https://raw.githubusercontent.com/fantasydatapros/LearnPythonWithFantasyFootball/master/2023/07-Data%20Visualizations/01-Weekly%20Fantasy%20Stats%20-%202022/weekly_df.csv')


# In[71]:


allen = weekly_df.loc[weekly_df['Player'] == 'Josh Allen']
mahomes = weekly_df.loc[weekly_df['Player'] == 'Patrick Mahomes']
wilson = weekly_df.loc[weekly_df['Player'] == 'Russell Wilson']


# In[72]:


sns.set_style('whitegrid') # setting style
plt.subplots(figsize=(10, 8)) # setting figure size
plt.plot(wilson['Week'], wilson['StandardFantasyPoints']) # first argument is x, second is y
plt.plot(mahomes['Week'], mahomes['StandardFantasyPoints'])
plt.plot(allen['Week'], allen['StandardFantasyPoints'])
plt.legend(['Wilson', 'Mahomes', 'Allen']) # setting legend in order of how we plotted things
plt.xlabel('Week')
plt.ylabel('Fantasy Points Scored')
plt.title('Wilson vs. Mahomes vs. Lamar - week by week Fantasy Performance', fontsize=16, fontweight='bold') # adjusting font size to 16px
plt.show() # show our visualization, not completely necessary, but supresses unneccessary output from matplotlib


# In[38]:


df = pd.read_csv('https://raw.githubusercontent.com/fantasydatapros/LearnPythonWithFantasyFootball/master/2023/07-Data%20Visualizations/03-Yearly%20Fantasy%20Stats%20-%202022.csv').iloc[:, 1:]


df['Usage/G'] = (df['PassingAtt'] + df['Tgt'] + df['RushingAtt'])/df['G']
df['FantasyPoints/G'] = df['FantasyPoints'] / df['G']
df.head()


# In[39]:


sns.lmplot(data=df, x='Usage/G', y='FantasyPoints/G', hue='Pos', height=7);


# In[40]:


combine_df = pd.read_csv("https://raw.githubusercontent.com/fantasydatapros/LearnPythonWithFantasyFootball/master/2023/07-Data%20Visualizations/02-Combine%20Data%202000%20to%202023.csv")
combine_df.head()


# In[41]:


plt.figure(figsize=(8, 8))


# In[42]:


sns.boxplot(x='Pos', y='40YD', data=combine_df.loc[combine_df['Pos'].isin(['RB', 'QB', 'TE', 'WR'])], palette=sns.color_palette("husl", n_colors=4));


# In[ ]:




