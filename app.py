import datetime
import operator
import pandas as pd
import numpy as np

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
  
pred_confirmed=[];
pred_death=[];
pred_recovered=[];    

pred_confirmed=[2380994.9,2455366.9,2531272.4,2608726.8,2687746.0,2768345.5,2850540.9,2934348.0,3019782.4,3106859.7];

pred_death=[83398.1,86003.3,88662.2,91375.4,94143.4,96966.8,99846.0,102781.8,105774.5,108824.8];

pred_recovered=[551611.4,568850.5,586445.1,604398.8,622715.1,641397.8,660450.3,679876.5,699679.9,719864.0];

from flask import Flask,render_template
app = Flask(__name__)
    
@app.route('/')
def student():
    return render_template('index.html')

@app.route('/confirmed')
def confirm(): 
    return render_template("confirmed.html",c=confirmed_dict,Date=future_forcast_dates[-10:],confirm=pred_confirmed);

@app.route('/death')
def death(): 
    return render_template("death.html",d=deaths_dict,Date=future_forcast_dates[-10:],death=pred_death);


@app.route('/recovered')
def recover(): 
    return render_template("recovered.html",r=recovered_dict,Date=future_forcast_dates[-10:],recover=pred_recovered);

if __name__ == '__main__':
    app.run()
      
