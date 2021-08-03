import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###### Import Space Weather data
#NOTE: This takes a WHILE to run!
month = ['01','02','03','04','05','06','07','08','09','10','11','12']
days = ['31','28','31','30','31','30','31','31','30','31','30','31']
sat = ['goes13','goes14','goes15']
year = [[2012,2013,2014,2015,2016,2017],[2018,2019,2020],[2012,2013,2014,2015,2016,2017,2018,2019,2020]]

sp = pd.DataFrame(columns=['T0','Long'])

for j in range(len(sat)):
    for k in range(len(year[j])):
        for i in range(len(month)):
            file = 'g'+sat[j][4:]+'_xrs_1m_'+str(year[j][k])+month[i]+'01_'+str(year[j][k])+month[i]+days[i]+'.csv'
            url = 'https://satdat.ngdc.noaa.gov/sem/goes/data/avg/'+str(year[j][k])+'/'+month[i]+'/'+str(sat[j])+'/csv/'+file
            print(url)
            temp = pd.read_csv(url,skiprows=150,names=['T0','Sample','Short','SampleL','Long'])
            
            temp = temp[['T0','Long']]
            temp.loc[temp['Long']==-99999] = None
            temp = temp.dropna()
            
            temp['T0'] = pd.to_datetime(temp['T0'])
            temp = temp.set_index('T0').groupby(pd.Grouper(freq='d')).mean().dropna(how='all')
            sp = sp.append(temp)

###### Import NOAA info
df = pd.read_excel(r'NOAA_Info\anom5j.xls')
pd.set_option('display.max_rows',10)

#Add in satcatalog data
sc = pd.read_csv('satcat.csv', header=None)

#Add in extra feature data
xf1 = pd.read_excel(r'Additional Features.xlsx')
xf2 = pd.read_excel(r'Additional Features.xlsx',sheet_name=1)

xf1.update(xf1['Name'].str.lower())
xf2.update(xf2['Name'].str.lower())

#Add in sunspot number data
ss = pd.read_csv('sunspot.csv')
ss['T0'] = pd.to_datetime(ss[['Month','Day','Year']])
ss = ss[['SSN','T0']]
ss = ss.dropna()

#Keep only the name, period, peri, apo, and incl
sc = sc[[3,20,22]]
sc.columns = ['Name','Perigee (km)','Inclination (deg)']
sc.update(sc['Name'].str.lower())

#Only keep satellites that have more than 1 anomalies
anom = df[df['BIRD'].isin(df['BIRD'].value_counts()[df['BIRD'].value_counts()>1].index)]
anom.update(anom['BIRD'].str.replace('-',' '))
anom.update(anom['BIRD'].str.lower())
lst = anom['BIRD'].unique()

#Select only the entries that have names in our anomaly list
newsc = sc[sc['Name'].isin(lst)]
lst2 = newsc['Name'].unique()
newanom = anom[anom['BIRD'].isin(lst2)]

#Keep only the bird and the ADate
newanom = newanom[['BIRD','ADATE']]

#Get the GPS data that we need
names = ['gps 5111','gps 5112','gps 5113','gps 5114','gps 5118','gps 9794','gps svn11']
gpsanom = anom[anom['BIRD'].isin(names)]

gpsanom = gpsanom[['BIRD','ADATE']]

#Add the GPS data and reset the index to get all the information
finalanom = pd.concat([newanom,gpsanom]) 
finalanom = finalanom.reset_index()
newsc = pd.concat([newsc,xf1[['Name','Perigee (km)','Inclination (deg)']]])

#Convert the time of anomaly to time between anomalies
data = pd.DataFrame(columns = ['BIRD','TTE','T0'])

for e in finalanom['BIRD'].unique():
    group = finalanom[finalanom['BIRD'].str.startswith(e, na=False)]
    group = group.reset_index()
    
    for i in range(len(group)-1):
        time = group['ADATE'].loc[i+1] - group['ADATE'].loc[i]
        data.loc[len(data)] = [e] + [time] + [group['ADATE'].loc[i]]
#NOTE: I have a lot of "0" data in the time, might want to deal with that somehow    

#Convert the timedelta to an integer and drop all the zero days
data['TTE'] = data['TTE'].dt.days
#NOTE: I am not dropping zero days to see how it pans out
data.loc[data['TTE']==0] = None
data = data.dropna()

#Drop all days that have "Time to Event" greater than 1 year
data.loc[data['TTE']>=365] = None
data = data.dropna()

#Drop all spacecraft that have fewer than "numanom" datapoints left
numanom = 20
data = data[data['BIRD'].isin(data['BIRD'].value_counts()[data['BIRD'].value_counts()>numanom].index)]

#Drop all datapoints from the Scatha spacecraft
data.loc[data['BIRD']=='scatha'] = None
data.loc[data['BIRD']=='ecs 1'] = None
data = data.dropna()

#Add sunspot number to all of the data entries
data = pd.merge(data,ss,on='T0')
data = pd.merge(data,sp,on='T0')

#Sort the data to be in ascending order by TTE
data = data.sort_values(['TTE'])
data = data.reset_index()
data = data.drop(['index'],axis=1)

lst3 = data['BIRD'].unique()
newsc = newsc[newsc['Name'].isin(lst3)]

#Add the mass values to the features
newsc = pd.merge(newsc,xf2, on='Name')

newsc = newsc.reset_index()
newsc = newsc.drop(['index'],axis=1)

#Add the starting month to the data
data['Month'] = data['T0'].dt.month

#Save the dataframes to a CSV file
data.to_csv(r'data.csv')
newsc.to_csv(r'satfeat.csv')
sp.to_csv(r'spaceweather.csv')