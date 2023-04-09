# -*- coding: utf-8 -*-
"""IPL_WIN_PREDICTOR.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1om74WNsJ1T0R0QEUCwKjtgqyKB-GcvfO
"""

import pandas as pd
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

matches=pd.read_csv('/content/drive/MyDrive/ipl/matches.csv')

deliveries=pd.read_csv('/content/drive/MyDrive/ipl/deliveries.csv')

matches.shape

matches.head()

deliveries.head()

total_score_df=deliveries.groupby(['id','inning']).sum()['total_runs'].reset_index()

total_score_df

total_score_df=total_score_df[total_score_df['inning']==1]

match_df=matches.merge(total_score_df[['id','total_runs']],left_on='id',right_on='id')

match_df

match_df['team1'].unique()

teams=['Royal Challengers Bangalore', 'Kings XI Punjab',
       'Mumbai Indians', 'Kolkata Knight Riders',
       'Rajasthan Royals','Chennai Super Kings','Sunrisers Hyderabad', 'Delhi Capitals']

match_df['team1']=match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2']=match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

match_df['team1']=match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2']=match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')

match_df=match_df[match_df['team1'].isin(teams)]
match_df=match_df[match_df['team2'].isin(teams)]

match_df['team1'].unique()

match_df.shape

match_df['neutral_venue'].value_counts()

match_df=match_df[match_df['neutral_venue']==0]

match_df

match_df=match_df[['id','city','winner','total_runs']]
match_df

delivery_df=match_df.merge(deliveries,left_on='id',right_on='id')

delivery_df

delivery_df=delivery_df[delivery_df['inning']==2]
delivery_df

delivery_df.shape

delivery_df['current_score']=delivery_df.groupby('id').cumsum()['total_runs_y']

delivery_df

delivery_df['runs_left']=delivery_df['total_runs_x']-delivery_df['current_score']
delivery_df

delivery_df['balls_left']=126-(delivery_df['over']*6+delivery_df['ball'])
delivery_df

delivery_df['is_wicket']=delivery_df['is_wicket'].astype('int')
wickets=delivery_df.groupby('id').cumsum()['is_wicket'].values
delivery_df['wickets_left']=10-wickets

delivery_df

delivery_df['current_run_rate'] = (delivery_df['current_score']*6)/(120 - delivery_df['balls_left'])

delivery_df.describe()



#required run rate
delivery_df['required_run_rate']=(delivery_df['runs_left']*6)/(delivery_df['balls_left'])
delivery_df

def result(row):
  return 1 if row['batting_team']==row['winner'] else 0



delivery_df['result']=delivery_df.apply(result,axis=1)
delivery_df

final_df=delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets_left','total_runs_x','current_run_rate','required_run_rate','result']]
final_df

delivery_df.describe()

delivery_df=delivery_df[delivery_df['balls_left']!=0]
delivery_df=delivery_df[delivery_df['balls_left']!=120]

delivery_df.describe()

final_df.describe()

final_df = final_df.sample(final_df.shape[0])

final_df.sample()

final_df.dropna(inplace=True)

final_df=final_df[final_df['balls_left']!=0]
final_df=final_df[final_df['balls_left']!=120]

final_df.describe()

final_df

final_df.isnull().sum()

X=final_df.iloc[:,:-1]
Y=final_df.iloc[:,-1]

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.2,random_state=1)

train_x

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])
]
,remainder='passthrough')

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe=Pipeline(steps=[
    ('step1',trf),
    ('step2',LogisticRegression(solver='liblinear'))
])

pipe.fit(train_x,train_y)

y_pred=pipe.predict(test_x)

from sklearn.metrics import accuracy_score
accuracy_score(test_y,y_pred)

pipe.predict_proba(test_x)[10]

def match_progression(x_df,match_id,pipe):
    match = x_df[x_df['id'] == match_id]
    match = match[(match['ball'] == 6)]
    temp_df = match[['batting_team','bowling_team','city','runs_left','balls_left','wickets_left','total_runs_x','current_run_rate','required_run_rate']].dropna()
    temp_df = temp_df[temp_df['balls_left'] != 0]
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0]*100,1)
    temp_df['win'] = np.round(result.T[1]*100,1)
    temp_df['end_of_over'] = range(1,temp_df.shape[0]+1)
    
    target = temp_df['total_runs_x'].values[0]
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0,target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    wickets = list(temp_df['wickets_left'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0,10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[0:temp_df.shape[0]]
    
    print("Target-",target)
    temp_df = temp_df[['end_of_over','runs_after_over','wickets_in_over','lose','win']]
    return temp_df,target

temp_df,target = match_progression(delivery_df,335987,pipe)
temp_df

import matplotlib.pyplot as plt
plt.figure(figsize=(18,8))
plt.plot(temp_df['end_of_over'],temp_df['wickets_in_over'],color='yellow',linewidth=3)
plt.plot(temp_df['end_of_over'],temp_df['win'],color='#00a65a',linewidth=4)
plt.plot(temp_df['end_of_over'],temp_df['lose'],color='red',linewidth=4)
plt.bar(temp_df['end_of_over'],temp_df['runs_after_over'])
plt.title('Target-' + str(target))

delivery_df['city'].unique()

import pickle
pickle.dump(pipe,open('pipe.pkl','wb'))