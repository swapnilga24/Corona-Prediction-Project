import matplotlib.pyplot as plt
import datetime
import operator
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV,train_test_split

#import datasets
confirmed_cases=pd.read_csv('Dataset/time_series_covid_19_confirmed.csv');
deaths_reported=pd.read_csv('Dataset/time_series_covid_19_deaths.csv');
recoveries_cases=pd.read_csv('Dataset/time_series_covid_19_recovered.csv');

cols=confirmed_cases.keys()

#select only dates from data. 
confirmed=confirmed_cases.loc[:,cols[4]:cols[-1]]
deaths=deaths_reported.loc[:,cols[4]:cols[-1]]
recovered=recoveries_cases.loc[:,cols[4]:cols[-1]]

#loop to calculate total corona cases,total deaths,total recovered and morality rate 
dates=confirmed.keys()
world_cases=[]
total_deaths=[]
total_recovered=[]
morality_rate=[]

for i in dates:
    confirmed_sum=confirmed[i].sum();
    death_sum=deaths[i].sum();
    recovered_sum=recovered[i].sum();
    world_cases.append(confirmed_sum);
    total_deaths.append(death_sum);
    total_recovered.append(recovered_sum);
    morality_rate.append(death_sum/confirmed_sum);
    
#print(confirmed_sum,death_sum,recovered_sum)

days_since_1_22=np.array([i for i in range(len(dates))]).reshape(-1,1)
world_cases=np.array(world_cases).reshape(-1,1)
total_deaths=np.array(total_deaths).reshape(-1,1)
total_recovered=np.array(total_recovered).reshape(-1,1)

days_in_future=10;
future_forecast=np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1,1)
adjust_dates=future_forecast[:-10]

unique_countries = list(confirmed_cases['Country/Region'].unique())

latest_confirmed=confirmed_cases[dates[-1]]
latest_recoveries=recoveries_cases[dates[-1]]
latest_deaths=deaths_reported[dates[-1]]

#Country corona cases total
country_confirmed_cases=[]
no_cases = []

for i in unique_countries:
    cases=latest_confirmed[confirmed_cases['Country/Region']==i].sum()
    if cases>0:
        country_confirmed_cases.append(cases);
    else:
        no_cases.append(i);        
        
for i in no_cases:
    unique_countries.remove(i)
    
unique_countries=[k for k,v in sorted(zip(unique_countries,country_confirmed_cases),key=operator.itemgetter(1),reverse=True)]
for i in range(len(unique_countries)):
    country_confirmed_cases[i]=latest_confirmed[confirmed_cases['Country/Region']==unique_countries[i]].sum()
    
#Country corona deaths cases total
country_deaths_cases=[]
no_cases = []

for i in unique_countries:
    cases=latest_deaths[deaths_reported['Country/Region']==i].sum()
    if cases>0:
        country_deaths_cases.append(cases);
    else:
        no_cases.append(i);        
        
for i in no_cases:
    unique_countries.remove(i)
    
unique_countries=[k for k,v in sorted(zip(unique_countries,country_deaths_cases),key=operator.itemgetter(1),reverse=True)]
for i in range(len(unique_countries)):
    country_deaths_cases[i]=latest_deaths[deaths_reported['Country/Region']==unique_countries[i]].sum()    
    
#Country corona recovered cases total
country_recover_cases=[]
no_cases = []

for i in unique_countries:
    cases=latest_recoveries[recoveries_cases['Country/Region']==i].sum()
    if cases>0:
        country_recover_cases.append(cases);
    else:
        no_cases.append(i);        
        
for i in no_cases:
    unique_countries.remove(i)
    
unique_countries=[k for k,v in sorted(zip(unique_countries,country_recover_cases),key=operator.itemgetter(1),reverse=True)]
for i in range(len(unique_countries)):
    country_recover_cases[i]=latest_recoveries[recoveries_cases['Country/Region']==unique_countries[i]].sum()    
    
#print("Confirmed cases by country/Region");
    
confirmed_dict={};
for i in range(len(unique_countries)):
    confirmed_dict[unique_countries[i]]=country_confirmed_cases[i]
#print(confirmed_dict);  
    
#print("Deaths cases by country/Region");
deaths_dict={};
for i in range(len(unique_countries)):
    deaths_dict[unique_countries[i]]=country_deaths_cases[i]
#print(deaths_dict);        
    
#print("Recovered cases by country/Region");
recovered_dict={};
for i in range(len(unique_countries)):
    recovered_dict[unique_countries[i]]=country_recover_cases[i]
#print(recovered_dict);        

start='1/22/2020'
start_date=datetime.datetime.strptime(start,'%m/%d/%Y')
future_forcast_dates=[]

for i in range(len(future_forecast)):
    future_forcast_dates.append((start_date+datetime.timedelta(days=i)).strftime('%m/%d/%Y'))


#x train for confirmed cases.
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, world_cases, test_size=0.15, shuffle=False)

kernel=['poly','sigmoid','rbf']
c=[0.01,0.1,1,10]
gamma=[0.01,0.1,1]
epsilon=[0.01,0.1,1]
shrinking=[True,False]
svm_grid={'kernel':kernel,'C':c,'gamma':gamma,'epsilon':epsilon,'shrinking':shrinking}

svm=SVR()
svm_search=RandomizedSearchCV(svm,svm_grid,scoring='neg_mean_squared_error',cv=3,return_train_score=True,n_jobs=-1,n_iter=40,verbose=1)
svm_search.fit(X_train_confirmed,y_train_confirmed)

svm_confirmed=svm_search.best_estimator_
pred_confirmed=svm_confirmed.predict(future_forecast)

svm_test_pred=svm_confirmed.predict(X_test_confirmed)
plt.plot(svm_test_pred)
plt.plot(y_test_confirmed)
plt.savefig('static/confirm1.jpg')

plt.figure(figsize=(20,12))
plt.plot(adjust_dates,world_cases)
plt.plot(future_forecast,pred_confirmed,linestyle='dashed',color='red')
plt.title('number of coronavirus cases over time',size=30)
plt.xlabel('Days since 1/22/2020',size=30)
plt.ylabel('number of cases',size=30)
plt.legend(['Confirmed case','svm prediction'])
plt.xticks(size=15)
plt.yticks(size=15)
plt.savefig('static/confirm2.jpg')
plt.show()

print('Confirmed Future Predictions for next 10 days in corona cases');
set(zip(future_forcast_dates[-10:],pred_confirmed[-10:]))

#x train for deaths cases.
    
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, total_deaths, test_size=0.15, shuffle=False)

kernel=['poly','sigmoid','rbf']
c=[0.01,0.1,1,10]
gamma=[0.01,0.1,1]
epsilon=[0.01,0.1,1]
shrinking=[True,False]
svm_grid={'kernel':kernel,'C':c,'gamma':gamma,'epsilon':epsilon,'shrinking':shrinking}

svm=SVR()
svm_search=RandomizedSearchCV(svm,svm_grid,scoring='neg_mean_squared_error',cv=3,return_train_score=True,n_jobs=-1,n_iter=40,verbose=1)
svm_search.fit(X_train_confirmed,y_train_confirmed)

svm_confirmed=svm_search.best_estimator_
pred_death=svm_confirmed.predict(future_forecast)

svm_test_pred=svm_confirmed.predict(X_test_confirmed)
plt.plot(svm_test_pred)
plt.plot(y_test_confirmed)
plt.savefig('static/dea1.jpg')

plt.figure(figsize=(20,12))
plt.plot(adjust_dates,total_deaths)
plt.plot(future_forecast,pred_death,linestyle='dashed',color='red')
plt.title('number of deaths cases over time',size=30)
plt.xlabel('Days since 1/22/2020',size=30)
plt.ylabel('number of deaths cases',size=30)
plt.legend(['Deaths case','svm prediction'])
plt.xticks(size=15)
plt.yticks(size=15)
plt.savefig('static/dea2.jpg')
plt.show()

print('Death Future Predictions for next 10 days in corona cases');
set(zip(future_forcast_dates[-10:],pred_death[-10:]))

#x train for Recover cases.

X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22,total_recovered, test_size=0.15, shuffle=False)

kernel=['poly','sigmoid','rbf']
c=[0.01,0.1,1,10]
gamma=[0.01,0.1,1]
epsilon=[0.01,0.1,1]
shrinking=[True,False]
svm_grid={'kernel':kernel,'C':c,'gamma':gamma,'epsilon':epsilon,'shrinking':shrinking}

svm=SVR()
svm_search=RandomizedSearchCV(svm,svm_grid,scoring='neg_mean_squared_error',cv=3,return_train_score=True,n_jobs=-1,n_iter=40,verbose=1)
svm_search.fit(X_train_confirmed,y_train_confirmed)

svm_confirmed=svm_search.best_estimator_
pred_recovered=svm_confirmed.predict(future_forecast)

svm_test_pred=svm_confirmed.predict(X_test_confirmed)
plt.plot(svm_test_pred)
plt.plot(y_test_confirmed)
plt.savefig('static/reco1.jpg')

plt.figure(figsize=(20,12))
plt.plot(adjust_dates,total_recovered)
plt.plot(future_forecast,pred_recovered,linestyle='dashed',color='red')
plt.title('number of recover cases over time',size=30)
plt.xlabel('Days since 1/22/2020',size=30)
plt.ylabel('number recover  cases',size=30)
plt.legend(['Recover case','svm prediction'])
plt.xticks(size=15)
plt.yticks(size=15)
plt.savefig('static/reco2.jpg')
plt.show()

print('Recover Future Predictions for next 10 in corona cases');
set(zip(future_forcast_dates[-10:],pred_recovered[-10:]))

from flask import Flask,render_template
app_main = Flask(__name__)
    
@app_main.route('/')
def student():
    return render_template('index.html')

@app_main.route('/confirmed')
def confirm(): 
    return render_template("confirmed.html",c=confirmed_dict,Date=future_forcast_dates[-10:],confirm=pred_confirmed[-10:]);

@app_main.route('/death')
def death(): 
    return render_template("death.html",d=deaths_dict,Date=future_forcast_dates[-10:],death=pred_death[-10:]);

@app_main.route('/recovered')
def recover(): 
    return render_template("recovered.html",r=recovered_dict,Date=future_forcast_dates[-10:],recover=pred_recovered[-10:]);

if __name__ == '__main__':
    app_main.run()
