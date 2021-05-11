# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 03:14:22 2020

@author: User
"""

#import glob
import pandas as pd
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from numpy import savetxt
import glob
import os
import re
import datetime
import seaborn as sns
from scipy import stats

from sklearn.metrics import mean_squared_error

date=[datetime.datetime(1980,1,1)+datetime.timedelta(days=x) for x in range (0,13948)]
df=pd.DataFrame()
df['date']=pd.to_datetime(date)
df=df.set_index('date')

isots=pd.read_csv('isotermas.csv',sep=';')
pp2000=pd.read_csv('Pp_pmgv2_2000v1.csv',sep=';')
pp2000=np.array(pp2000)
a=len(pp2000)
S=(a,3)
S1=(a,15)
tormentas=np.zeros((a,5))
isotermasS=np.zeros(S)
p1=[0,0]
p2=[7500,7500]
x_val=[p1[0], p2[0]]
y_val=[p1[1],p2[1]]
cotas_polyfit=np.zeros(2)+(1,7500)
sl=np.zeros(S1)


umbral=[5,10,20,30,40]
for k in range(0,len(pp2000)-2):
    for j in range(0,len(umbral)): 
        r=umbral[j]
        if (pp2000[k,1]>r):
           tormentas[k,j]=pp2000[k,1]
#    if (pp2000[k,1]+pp2000[k+1,1]>5 and pp2000[k+1,1]>0):
#       tormentas[k+1,1]=pp2000[k,1]+pp2000[k+1,1]
#    if (pp2000[k,1]+pp2000[k+1,1]+ pp2000[k+2,1]>5 and pp2000[k+1,1]>0):
#       tormentas[k+1,2]=pp2000[k,1]+pp2000[k+1,1]+pp2000[k+2,1] #tormentas son las tormentas por días
#    if (pp2000[k,1]>5 and pp2000[k+1,1]>5 and pp2000[k+2,1]>5 and pp2000[k+3,1]>5):
#       tormentas[k,4]=pp2000[k,1]+pp2000[k+1,1]+pp2000[k+2,1]+pp2000[k+3,1]
    #else:
#t_tormentas.insert(0,'date',df,True)
#t_tormentas=t_tormentas.replace(0,np.nan).dropna()
isots=isots.values
'correccion isotermas'
for k in range(0,len(pp2000)):
    for j in range(3,6):
        if (580<isots[k,j] and isots[k,j]<6000):
            isotermasS[k,j-3]=isots[k,j]
       
'Identifica Lineas de nieve'
for k in range(0,len(isots)):
    m=0  
    for j in range(0,5): #umbral de lluvias
        if (tormentas[k,j]>0):
           for i in range(0,3):#isotermas
               sl[k,m]=isotermasS[k,i]
               if sl[k,m]<0:
                   sl[k,m]=0
               m=m+1
sl=pd.DataFrame(sl[0:,0:], columns=['I0-PMG-5mm', 'I1-PMG-5mm', 'I2-PMG-5mm','I0-PMG-10mm','I1-PMG-10mm','I2-PMG-10mm','I0-PMG-20mm','I1-PMG-20mm','I2-PMG-20mm','I0-PMG-30mm','I1-PMG-30mm','I2-PMG-30mm','I0-PMG-40mm','I1-PMG-40mm','I2-PMG-40mm'],index=df.index)

sl=sl.replace(0,np.nan)

#%%
ruta='H:/Natalia/AMTC/GloboSonda/diarios'
files=glob.glob(os.path.join(ruta,'GS-*'))
r=re.compile(r"(\d+)")
sfiles=sorted(files, key=lambda x: int(r.search(x).group()))
#print (sfiles)
a=int(len(sfiles)/2)
S1=(5,len(sfiles))
s=(a,5)
s1=(a,3)
isogs0=np.zeros(S1)
isogs12=np.zeros(S1)
cotags=np.zeros(S1)
tempgs=np.zeros(S1)
cota0gs=np.zeros(s)
temp0gs=np.zeros(s)
gs_int0=np.zeros(s1)
gs_int12=np.zeros(s1)
cota12gs=np.zeros(s)
temp12gs=np.zeros(s)
m=0
n=0
p=0
for file in sfiles:
    data=pd.read_csv(file,sep=';',header=2, index_col=None, usecols=[2,3])
    data=data.values
    temp=data[:,1]
    cota=data[:,0]
    b=len(data)
    for j in range(2,b-2):
        if(temp[j-1]*temp[j]<0 and cota[j]<7000):
            cotags[0,m]=cota[j-2]            
            cotags[1,m]=cota[j-1]
            cotags[2,m]=cota[j]    
            cotags[3,m]=cota[j+1]
            cotags[4,m]=cota[j+2]  
            tempgs[0,m]=temp[j-2]
            tempgs[1,m]=temp[j-1]  
            tempgs[2,m]=temp[j]    
            tempgs[3,m]=temp[j+1]
            tempgs[4,m]=temp[j+2]                     
    m=m+1
#%%
s2=5,m
s3=3,m
dT=np.zeros(s2)
gs_iso=np.array([0,1,2])  
gs_int=np.zeros(s3)
z=np.zeros(s3)
for k in range(0,m):
    for v in range(0,3): 
        for j in range(0,4):
            if gs_iso[v]<tempgs[j,k] and tempgs[j+1,k]<gs_iso[v]:
                dT=tempgs[j,k]-tempgs[j+1,k]
                dz=cotags[j,k]-cotags[j+1,k]
                dTi=dT-gs_iso[v]
                gs_int[v,k]=dz/dT*gs_iso[v]+cotags[j+1,k]
                

for k in range(0,m):
    if (k%2==0):
        temp0gs[n,:]=tempgs[:,k]
        cota0gs[n,:]=cotags[:,k]
        gs_int0[n,:]=gs_int[:,k]
        n=n+1
    else:
        temp12gs[p,:]=tempgs[:,k]
        cota12gs[p,:]=cotags[:,k]
        gs_int12[p,:]=gs_int[:,k]
        p=p+1

fechas0=[]
fechas12=[]
for i,q in enumerate(files):
    name=os.path.splitext ("H:\\Natalia\\AMTC\\GloboSonda\\diarios\\"+q[-20:])[0]
    date_string = '-'.join(name.split("-")[1:])
    ra=datetime.datetime.strptime(date_string, "%Y-%m-%d-%H")
    if (i%2==0):
        fechas0.append(ra)
    else:
        fechas12.append(ra)
'valores no interpolados'    
t_cota12=pd.DataFrame(data=cota12gs[0:,0:], columns=['H0', 'H1', 'H2','H3','H4'],index=fechas12)#=[datetime.datetime.strptime(date_string = '-'.join(q.split("-")[1:]), "%Y-%m-%d-%H").index[0] ]#, 'Lluvia 4'])
t_cota0=pd.DataFrame(data=cota0gs[0:,0:], columns=['H0', 'H1', 'H2','H3','H4'],index=fechas0)
'valores interpolados'
t_gs_int12=pd.DataFrame(data=gs_int12[0:,0:], columns=['I0-GS', 'I1-GS', 'I2-GS'],index=fechas12)#=[datetime.datetime.strptime(date_string = '-'.join(q.split("-")[1:]), "%Y-%m-%d-%H").index[0] ]#, 'Lluvia 4'])
t_gs_int0=pd.DataFrame(data=gs_int0[0:,0:], columns=['I0-GS', 'I1-GS', 'I2-GS'],index=fechas0)
t_gs_int12=t_gs_int12.replace(0,np.nan)
t_gs_int0=t_gs_int0.replace(0,np.nan)



t_gs_int0=t_gs_int0.merge(sl, how='outer',left_index=True,right_index=True)
t_gs_int12=t_gs_int12.merge(sl, how='outer',left_index=True,right_index=True)
#l,m=cotags.shape #Tamaño de la matriz
#n,o=sl.shape

#%% 'Lineas de tendencia'
m=0

egs=np.zeros((15,4))   
egs0=np.zeros((15,4))   
for i in range(0,len(umbral)):   
    for j in range(0,3): #isotermas
        name='I'+str(j)+'-PMG-'+str(umbral[i])+'mm'
        y=t_gs_int12[name]
        y0=t_gs_int0[name]
        x=t_gs_int12["I0-GS"] # 
        x0=t_gs_int0["I0-GS"] # 
        nas= np.logical_or(np.isnan(x),np.isnan(y))
        nas0= np.logical_or(np.isnan(x0),np.isnan(y0))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[~nas],y[~nas])
        slope0, intercept0, r_value0, p_value0, std_err0 = stats.linregress(x0[~nas0],y0[~nas0])
        egs[m,0]=slope
        egs[m,1]=intercept
        egs[m,2]=r_value
        egs[m,3]=mean_squared_error(x[~nas],y[~nas])
        egs0[m,0]=slope0
        egs0[m,1]=intercept0
        egs0[m,2]=r_value0
        egs0[m,3]=mean_squared_error(x0[~nas0],y0[~nas0])
        m=m+1 
"""np.savetxt('Estadisticos_GS12',egs, delimiter='&') 
np.savetxt('Estadisticos_GS00',egs0, delimiter='&') """
'Compara globos sonda con PMG'

sns.set(font_scale=1.5)
fig, axs = plt.subplots(2, 5,figsize=(18, 10), sharey=True)
#fig.suptitle('Globo Sonda ')
for i in range(0,len(umbral)): # Escenarios
    for j in range(0,2): #isotermas
        name='I'+str(j)+'-PMG-'+str(umbral[i])+'mm'
        ax1=sns.scatterplot(ax=axs[j,i],x='I0-GS',y=name,data=t_gs_int12, label='Isoterma'+ str(j)+'- 12UTC').set_title(str(umbral[i])+'mm',fontsize=18)
        ax2=sns.scatterplot(ax=axs[j,i],x='I0-GS',y=name,data=t_gs_int0, label='Isoterma'+ str(j)+'- 00UTC').set_title(str(umbral[i])+'mm',fontsize=18)
        axs[j,i].plot(x_val,y_val,'-',color='black')
        plt.tight_layout(pad=1, w_pad=1, h_pad=0.5)
        for ax in axs.flat:
            ax.set_xlabel('Isoterma G-S (m.s.n.m.)', fontsize=18)
            ax.set_ylabel('Isoterma PMG (m.s.n.m.)', fontsize=18)
            ax.set_xlim(0,7000)
            ax.set_ylim(0,7000)
            ax.legend(fontsize=14,ncol=1,loc=2)
        for ax in axs.flat:
            ax.label_outer()

"""fig
plt.title("1 día de lluvia-Comparación para 0-UTC")
ax=sns.scatterplot(x='I0-GS',y='1d-I0-PMG',data=t_gs_int0, label='Isoterma 0').set_title('lluvia 5mm')
ax=sns.scatterplot(x='I1-GS',y='1d-I1-PMG',data=t_gs_int0, label='Isoterma 0').set_title('lluvia 5mm')
ax=sns.scatterplot(x='I2-GS',y='1d-I2-PMG',data=t_gs_int0, label='Isoterma 0').set_title('lluvia 5mm')
ax=plt.plot(x_val, y_val,color='blue')
ax=plt.plot(x_val,u,color='black')
plt.xlabel('Globo Sonda m.s.n.m.')
plt.ylabel('PMG m.s.n.m')
plt.title("1 días de lluvia")"""
"""
fig, axs = plt.subplots(1, 3)
for j in range(0,3):
        # nas = np.logical_or(np.isnan(x),np.isnan(y))
         #slope, intercept, r_value, p_value, std_err = stats.linregress(x[~nas],y[~nas])
      #  u=slope*cotas_polyfit+intercept
        #fig.suptitle('LN para umbral 5mm'+ str(i-1)+'°C',fontsize=10)
        axs[0, j].t_gs_int0.plot(kind='scatter', x='H0-GS', y='I0-PMG', '.',color='black',markersize=4)
        axs[0, j].plot(x_val,y_val,'-',color='blue')
        axs[0, j].set_title('Isoterma'+str(j)+'-LLuvia'+str(j)+'dias',fontsize=8)
        for ax in axs.flat:
            ax.set_xlabel('LN Globo Sonda (m.s.n.m.)', fontsize=8)
            ax.set_ylabel('LN PMG (m.s.n.m.)', fontsize=8)
            ax.set_xlim(0,7500)
            ax.set_ylim(0,7500)
        for ax in axs.flat:
            ax.label_outer()                                        
#t_isogs=pd.DataFrame(data=isogs[0:,1:],columns=['0C','1C','2C'])
 """        


