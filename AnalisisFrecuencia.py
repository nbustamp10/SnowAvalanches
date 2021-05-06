# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:22:54 2020

@author: User
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from numpy import savetxt
import os
import re
import datetime
import seaborn as sns
from scipy import stats
import scipy.stats as st
from scipy.stats import skew
from scipy.stats import gumbel_r
from scipy.stats import pearson3
from scipy.stats import chisquare


date=[datetime.datetime(1980,1,1)+datetime.timedelta(days=x) for x in range (0,13948)]
df=pd.DataFrame()
df['date']=pd.to_datetime(date)
df=df.set_index('date')
#df['year'] =pd.DatetimeIndex(df['date']).year
#df=df.set_index('year')

#isots=pd.read_csv('isotermas.csv',sep=';')
pp=pd.read_csv('Pp_pmgmv2_zc1.csv',sep=';')
temp=pd.read_csv('Temp_pmgm_zc1.csv',sep=';')
pp=np.array(pp)
temp=np.array(temp)
temp=temp[:,1:]
a=len(pp)
S=(a,18)
S2=(39,18)
S1=(a,9)
tormentas1=np.zeros(S)
tormentas2=np.zeros(S)
tormentas3=np.zeros(S)
tormentas4=np.zeros(S)
tormentas5=np.zeros(S)
snow1=np.zeros(S)
snow2=np.zeros(S)
snow3=np.zeros(S)
snow4=np.zeros(S)
snow5=np.zeros(S)

isotermasS=np.zeros(S)
p1=[0,0]
p2=[7500,7500]
x_val=[p1[0], p2[0]]
y_val=[p1[1],p2[1]]
cotas_polyfit=np.zeros(2)+(1,7500)
sl=np.zeros(S1)
for k in range(0,len(pp)-4):
    for j in range(1,19):
        if (pp[k,j]>0 and pp[k+1,j]==0 and pp[k-1,j]==0):
           tormentas1[k,j-1]=pp[k,j]
        else:
           if ((pp[k,j]*pp[k+1,j])>0 and pp[k+1,j]>0 and pp[k+2,j]==0 and pp[k-1,j]==0):
               tormentas2[k,j-1]=pp[k,j]+pp[k+1,j]
           else:
               if ((pp[k,j]*pp[k+1,j]*pp[k+2,j])>0 and pp[k+3,j]==0 and pp[k-1,j]==0):
                   tormentas3[k,j-1]=pp[k,1]+pp[k+1,j]+pp[k+2,j]
               else:
                   if ((pp[k,j]*pp[k+1,j]*pp[k+2,j]*pp[k+3,j])>0 and pp[k+4,j]==0 and pp[k-1,j]==0):
                       tormentas4[k,3]=pp[k,j]+pp[k+1,j]+pp[k+2,j]+pp[k+3,j]
                   else:
                       if ((pp[k,j]*pp[k+1,j]*pp[k+2,j]*pp[k+3,j]*pp[k+4,j])>0 and pp[k+5,j]==0 and pp[k-1,j]==0):
                           tormentas5[k,j-1]=pp[k,j]+pp[k+1,j]+pp[k+2,j]+pp[k+3,j]+pp[k+4,j]
t_tormenta1=pd.DataFrame(tormentas1[0:,0:], columns=['ZC1', 'ZC2', 'ZC3','ZC4','ZC5','ZC6','ZC7','ZC8','ZC9','ZC10','ZC11','ZC12','ZC13','ZC14','ZC15','ZC16','ZC17','ZC18'],index=df.index)
t_tormenta2=pd.DataFrame(tormentas2[0:,0:], columns=['ZC1', 'ZC2', 'ZC3','ZC4','ZC5','ZC6','ZC7','ZC8','ZC9','ZC10','ZC11','ZC12','ZC13','ZC14','ZC15','ZC16','ZC17','ZC18'],index=df.index)
t_tormenta3=pd.DataFrame(tormentas3[0:,0:], columns=['ZC1', 'ZC2', 'ZC3','ZC4','ZC5','ZC6','ZC7','ZC8','ZC9','ZC10','ZC11','ZC12','ZC13','ZC14','ZC15','ZC16','ZC17','ZC18'],index=df.index)
t_tormenta4=pd.DataFrame(tormentas4[0:,0:], columns=['ZC1', 'ZC2', 'ZC3','ZC4','ZC5','ZC6','ZC7','ZC8','ZC9','ZC10','ZC11','ZC12','ZC13','ZC14','ZC15','ZC16','ZC17','ZC18'],index=df.index)
t_tormenta5=pd.DataFrame(tormentas5[0:,0:], columns=['ZC1', 'ZC2', 'ZC3','ZC4','ZC5','ZC6','ZC7','ZC8','ZC9','ZC10','ZC11','ZC12','ZC13','ZC14','ZC15','ZC16','ZC17','ZC18'],index=df.index)

for k in range(0,len(pp)-4):
    for j in range(0,18):
        if (tormentas1[k,j]>0 and (temp[k,j]<1 or temp[k+1,j]<1)):
            snow1[k,j]=tormentas1[k,j]
        if (tormentas2[k,j]>0 and (temp[k,j]<1 or temp[k+1,j]<1)):
            snow2[k,j]=tormentas2[k,j]
        if (tormentas3[k,j]>0 and (temp[k,j]<1 or temp[k+1,j]<1)):
            snow3[k,j]=tormentas3[k,j]
        if (tormentas4[k,j]>0 and (temp[k,j]<1 or temp[k+1,j]<1)):
            snow4[k,j]=tormentas4[k,j]
        if (tormentas5[k,j]>0 and (temp[k,j]<1 or temp[k+1,j]<1)):
            snow5[k,j]=tormentas5[k,j]
t_snow1=pd.DataFrame(snow1[0:,0:], columns=['ZC1', 'ZC2', 'ZC3','ZC4','ZC5','ZC6','ZC7','ZC8','ZC9','ZC10','ZC11','ZC12','ZC13','ZC14','ZC15','ZC16','ZC17','ZC18'],index=df.index)
t_snow2=pd.DataFrame(snow2[0:,0:], columns=['ZC1', 'ZC2', 'ZC3','ZC4','ZC5','ZC6','ZC7','ZC8','ZC9','ZC10','ZC11','ZC12','ZC13','ZC14','ZC15','ZC16','ZC17','ZC18'],index=df.index)
t_snow3=pd.DataFrame(snow3[0:,0:], columns=['ZC1', 'ZC2', 'ZC3','ZC4','ZC5','ZC6','ZC7','ZC8','ZC9','ZC10','ZC11','ZC12','ZC13','ZC14','ZC15','ZC16','ZC17','ZC18'],index=df.index)
t_snow4=pd.DataFrame(snow4[0:,0:], columns=['ZC1', 'ZC2', 'ZC3','ZC4','ZC5','ZC6','ZC7','ZC8','ZC9','ZC10','ZC11','ZC12','ZC13','ZC14','ZC15','ZC16','ZC17','ZC18'],index=df.index)
t_snow5=pd.DataFrame(snow5[0:,0:], columns=['ZC1', 'ZC2', 'ZC3','ZC4','ZC5','ZC6','ZC7','ZC8','ZC9','ZC10','ZC11','ZC12','ZC13','ZC14','ZC15','ZC16','ZC17','ZC18'],index=df.index)


'Máximo anual precipitación (ma)'
ma_1=t_tormenta1.resample('Y').max()
ma_2=t_tormenta2.resample('Y').max()
ma_3=t_tormenta3.resample('Y').max()
ma_4=t_tormenta4.resample('Y').max()
ma_5=t_tormenta5.resample('Y').max()
'Máximo anual de nieve (mas)'
mas_1=t_snow1.resample('Y').max()
mas_2=t_snow2.resample('Y').max()
mas_3=t_snow3.resample('Y').max()
mas_4=t_snow4.resample('Y').max()
mas_5=t_snow5.resample('Y').max() 
'Ordena de mayor a menor'

ma_1s=pd.concat([ma_1[col].sort_values(ascending=False).reset_index(drop=True) for col in ma_1], axis=1, ignore_index=True)
ma_2s=pd.concat([ma_2[col].sort_values(ascending=False).reset_index(drop=True) for col in ma_2], axis=1, ignore_index=True)
ma_3s=pd.concat([ma_3[col].sort_values(ascending=False).reset_index(drop=True) for col in ma_3], axis=1, ignore_index=True)
ma_4s=pd.concat([ma_4[col].sort_values(ascending=False).reset_index(drop=True) for col in ma_4], axis=1, ignore_index=True)
ma_5s=pd.concat([ma_5[col].sort_values(ascending=False).reset_index(drop=True) for col in ma_5], axis=1, ignore_index=True)
 
mas_1s=pd.concat([mas_1[col].sort_values(ascending=False).reset_index(drop=True) for col in mas_1], axis=1, ignore_index=True)
mas_2s=pd.concat([mas_2[col].sort_values(ascending=False).reset_index(drop=True) for col in mas_2], axis=1, ignore_index=True)
mas_3s=pd.concat([mas_3[col].sort_values(ascending=False).reset_index(drop=True) for col in mas_3], axis=1, ignore_index=True)
mas_4s=pd.concat([mas_4[col].sort_values(ascending=False).reset_index(drop=True) for col in mas_4], axis=1, ignore_index=True)
mas_5s=pd.concat([mas_5[col].sort_values(ascending=False).reset_index(drop=True) for col in mas_5], axis=1, ignore_index=True)
mas_1s=mas_1s.replace(0,np.nan)
mas_2s=mas_2s.replace(0,np.nan)
mas_3s=mas_3s.replace(0,np.nan)
mas_1s.to_csv('MaxA-nieve-1d', sep=';')
mas_2s.to_csv('MaxA-nieve-2d', sep=';')
mas_3s.to_csv('MaxA-nieve-3d', sep=';')
mas_4s.to_csv('MaxA-nieve-4d', sep=';')
mas_5s.to_csv('MaxA-nieve-5d', sep=';')
'Media'
mean_mas_1=mas_1s.mean()
mean_mas_2=mas_2s.mean()
mean_mas_3=mas_3s.mean()

'Standar desviation'
std_mas_1=mas_1s.std() 
std_mas_2=mas_2s.std() 
std_mas_3=mas_3s.std() 
#std_mas_4=mas_4s.std() 
#std_mas_5=mas_5s.std() 
'Varianza'
var_mas_1=mas_1s.var()
var_mas_2=mas_2s.var()
var_mas_3=mas_3s.var()
'Logartimos'
mean_log_mas1=np.log(mas_1s).mean()
mean_log_mas2=np.log(mas_2s).mean()
mean_log_mas3=np.log(mas_3s).mean()

std_log_mas1=np.log(mas_1s).std()
std_log_mas2=np.log(mas_2s).std()
std_log_mas3=np.log(mas_3s).std()
#%%
'Con los datos'
a=len(mas_1s)
S2=(39,18)

lognormal=np.zeros(S2)
pearson=np.zeros(S2)
logpearson=np.zeros(S2)
gumbel=np.zeros(S2)
normal=np.zeros(S2)
pf=np.zeros(a)
zf=np.zeros(a)
zf1=np.zeros(a)

w=np.zeros(S2)
tr=np.zeros(S2)
b=np.zeros(18)
C=np.zeros(18)
se=np.zeros(18)
cs=np.zeros(18)
yn1=np.zeros(18)
sigman1=np.zeros(18)
p=np.zeros(S2)
z=np.zeros(S2)
ktd=np.zeros(S2)
ktp=np.zeros(S2)
ktlp=np.zeros(S2)
chisq=np.zeros((4,18))
ch2=np.zeros((4,18))

for j in range(0,18):
    b[j]=np.count_nonzero(np.isnan(mas_2s[j]))
    for k in range(0,a):
        auxiliar=mas_2s[j].values
        c=np.argwhere(np.isnan(auxiliar)).min()
        C[j]=c
        tr[k,j]=(a-b[j]+1)/(k+1)
        p[k,j]=1/tr[k,j]
        #ktd[k,j]=gumbel_r.ppf(1-p[k,j]) #Factor de frecuencia de la Gumbel
        if (c>=20):
            if (c<=25):
                yn=(0.5309-0.5236)/5*(c-20)+0.5236
                sigman=(1.0914-1.0628)/5*(c-20)+1.0628
            if (c>25 and c<=30):
                yn=(0.5362-0.5309)/5*(c-25)+0.5309
                sigman=(1.1124-1.0914)/5*(c-25)+1.0914
            if (c>30 and c<=35):
                yn=(0.5403-0.5362)/5*(c-30)+0.5362
                sigman=(1.1285-1.1124)/5*(c-30)+1.1124
            if (c>35 and c<=40):
                yn=(0.5436-0.5403)/5*(c-35)+0.5403
                sigman=(1.1413-1.1285)/5*(c-35)+1.1285
            yn1[j]=yn
            sigman1[j]=sigman
            se[j]=skew(auxiliar[:c])#sesgo
            cs[j]=(c/((c-1)*(c-2)))*(((np.log(auxiliar[:c])-mean_log_mas2[j])/std_log_mas2[j])**3).sum()
            ktp[k,j]=pearson3.ppf([1-p[k,j]], se[j]) #Factor de frecuencia de Pearson3
            z[k,j]=st.norm.ppf(1-p[k,j]) #K de variable normal estandalizada
            ktd[k,j]=(-np.log(np.log(tr[k,j]/(tr[k,j]-1)))-yn)/sigman #Kt-Gumbel
            ktlp[k,j]=z[k,j]+(z[k,j]**2-1)*(cs[j]/6)+1/3*(z[k,j]**3-6*z[k,j])*((cs[j]/6)**2)-(z[k,j]**2-1)*(cs[j]/6)**3+z[k,j]*(cs[j]/6)**4+1/3*(cs[j]/6)**5
            #if (p[k,j]<0.5):
             #   w[k,j]=(np.log(1/p[k,j]**2))**(0.5)
                #z[k,j]=w[k,j]-(2.515517+0.802853*w[k,j]+0.010328*w[k,j]**2)/(1+1.432788*w[k,j]+0.1889269*w[k,j]**2+0.001308*w[k,j]**3)
            #else:
             #   w[k,j]=(np.log(1/(1-p[k,j])**2))**(0.5)
                #z[k,j]=-w[k,j]+(2.515517+0.802853*w[k,j]+0.010328*w[k,j]**2)/(1+1.432788*w[k,j]+0.1889269*w[k,j]**2+0.001308*w[k,j]**3)
            #ktd[k,j]=(-np.log(np.log(tr[k,j]/(tr[k,j]-1)))-0.54)/1.134 #Kt Gumbel
            gumbel[k,j]=mean_mas_2[j]+ktd[k,j]*std_mas_2[j]
            normal[k,j]=mean_mas_2[j]+z[k,j]*std_mas_2[j]
            lognormal[k,j]=np.exp(mean_log_mas2[j]+std_log_mas2[j]*z[k,j])
            pearson[k,j]=mean_mas_2[j]+ktp[k,j]*std_mas_2[j]
            logpearson[k,j]=np.exp(mean_log_mas2[j]+std_log_mas2[j]*ktlp[k,j])
            'Bondad de ajutste'
            chisq[0,j]=chisquare(auxiliar[:c],normal[:c,j])[1]       #Bondad de ajuste para la normal
            chisq[1,j]=chisquare(auxiliar[:c],lognormal[:c,j])[1]    #Bondad de ajuste para la lognormal
            chisq[2,j]=chisquare(auxiliar[:c],gumbel[:c,j])[1]       #Bondad de ajuste para la normal
            chisq[3,j]=chisquare(auxiliar[:c],pearson[:c,j])[1]      #Bondad de ajuste para la normal
        for i in range(0,4):
            if (chisq[i,j]<=0.05):
                ch2[i,j]=0 #rechazo
            else:
                ch2[i,j]=1 #acepta
        #logpearson[k,j]=np.exp(mean_log_mas1[j]+std_log_mas1[j]*ktlp[k,j])
"""
np.savetxt('Gumbel1D_weibull', gumbel, delimiter=';')
np.savetxt('Normal1D_weibull', normal, delimiter=';')
np.savetxt('Lognormal1D_weibull', lognormal,delimiter=';') 
np.savetxt('Pearson1d_weibull', pearson,delimiter=';') 
np.savetxt('Tr_weibull1d', tr,delimiter=';') 
np.savetxt('Chi2_weibull1d',ch2,delimiter=';')"""    

'Analisis para periodos posteriores'
trf=[1.5,2,5,10,25,30,50,100,200,1000]
S1=(len(trf),len(mean_mas_1))
normal1=np.zeros(S1)
normal2=np.zeros(S1)
normal3=np.zeros(S1)
nor_sup1=np.zeros(S1)
nor_inf1=np.zeros(S1)
nor_sup2=np.zeros(S1)
nor_inf2=np.zeros(S1)
nor_sup3=np.zeros(S1)
nor_inf3=np.zeros(S1)
gum_sup1=np.zeros(S1)
gum_sup2=np.zeros(S1)
gum_sup3=np.zeros(S1)
gum_inf1=np.zeros(S1)
gum_inf2=np.zeros(S1)
gum_inf3=np.zeros(S1)
pear_sup1=np.zeros(S1)
pear_sup2=np.zeros(S1)
pear_sup3=np.zeros(S1)
pear_inf1=np.zeros(S1)
pear_inf2=np.zeros(S1)
pear_inf3=np.zeros(S1)
logn_sup1=np.zeros(S1)
logn_inf1=np.zeros(S1)
lognormal1=np.zeros(S1)
lognormal2=np.zeros(S1)
lognormal3=np.zeros(S1)
kta=np.zeros(S1)
ktb=np.zeros(S1)
gumbel1=np.zeros(S1)
gumbel2=np.zeros(S1)
gumbel3=np.zeros(S1)
pearson1=np.zeros(S1)
pearson2=np.zeros(S1)
pearson3d=np.zeros(S1)
logpearson1=np.zeros(S1)
logpearson2=np.zeros(S1)
logpearson3d=np.zeros(S1)
w1=np.zeros(len(trf))
ktg=np.zeros(S1)
ktp1=np.zeros(S1)
ktlp1=np.zeros(S1)

for j in range(0,len(mean_mas_2)):
    for i in range(0,len(trf)):
        t=trf[i]
        pf[i]=1/t
        zf[i]=st.norm.ppf(1-pf[i])
        #ktg[i]=gumbel_r.ppf(1-pf[i])
        ktg[i,j]=(-np.log(np.log(t/(t-1)))-yn1[j])/sigman1[j]
        ktp1[i,j]=pearson3.ppf([1-pf[i]], se[j])
        ktlp1[i,j]=zf[i]+(zf[i]**2-1)*(cs[j]/6)+1/3*(zf[i]**3-6*zf[i])*((cs[j]/6)**2)-(zf[i]**2-1)*(cs[j]/6)**3+zf[i]*(cs[j]/6)**4+1/3*(cs[j]/6)**5
        #if (pf[i]<0.5):
         #   w1[i]=np.log(1/pf[i]**2)**(0.5)
            #zf[i]=w1[i]-(2.515517+0.802853*w1[i]+0.010328*w1[i]**2)/(1+1.432788*w1[i]+0.1889269*w1[i]**2+0.001308*w1[i]**3)
        #else:
         #   w1[i]=np.log(1/(1-pf[i])**2)**(0.5)
            #zf[i]=-w1[i]+(2.515517+0.802853*w1[i]+0.010328*w1[i]**2)/(1+1.432788*w1[i]+0.1889269*w1[i]**2+0.001308*w1[i]**3)
        normal1[i,j]=mean_mas_1[j]+zf[i]*std_mas_1[j]
        normal2[i,j]=mean_mas_2[j]+zf[i]*std_mas_2[j]
        normal3[i,j]=mean_mas_3[j]+zf[i]*std_mas_3[j]
        lognormal1[i,j]=np.exp(mean_log_mas1[j]+std_log_mas1[j]*zf[i])
        lognormal2[i,j]=np.exp(mean_log_mas2[j]+std_log_mas2[j]*zf[i])
        lognormal3[i,j]=np.exp(mean_log_mas3[j]+std_log_mas3[j]*zf[i])
        #ktg[i]=-(6)**(0.5)/np.pi*(0.5772+np.log(np.log(t/(t-1)))) #Kt-Gumbel
        gumbel1[i,j]=mean_mas_1[j]+ktg[i,j]*std_mas_1[j]
        gumbel2[i,j]=mean_mas_2[j]+ktg[i,j]*std_mas_2[j]
        gumbel3[i,j]=mean_mas_3[j]+ktg[i,j]*std_mas_3[j]
        pearson1[i,j]=mean_mas_1[j]+ktp1[i,j]*std_mas_1[j]
        pearson2[i,j]=mean_mas_2[j]+ktp1[i,j]*std_mas_2[j]
        pearson3d[i,j]=mean_mas_3[j]+ktp1[i,j]*std_mas_3[j]
        logpearson1[i,j]=np.exp(mean_log_mas1[j]+std_log_mas1[j]*ktlp1[i,j])
        logpearson2[i,j]=np.exp(mean_log_mas2[j]+std_log_mas2[j]*ktlp1[i,j])
        logpearson3d[i,j]=np.exp(mean_log_mas3[j]+std_log_mas3[j]*ktlp1[i,j])
        #intervalos de confianza Normal
        nor_sup1[i,j]=normal1[i,j]+std_mas_1[j]/C[j]**(0.5)*(1+zf[i]**2/2)**(0.5)*st.norm.ppf(0.95)
        nor_inf1[i,j]=normal1[i,j]-std_mas_1[j]/C[j]**(0.5)*(1+zf[i]**2/2)**(0.5)*st.norm.ppf(0.95)
        nor_sup2[i,j]=normal2[i,j]+std_mas_2[j]/C[j]**(0.5)*(1+zf[i]**2/2)**(0.5)*st.norm.ppf(0.95)
        nor_inf2[i,j]=normal2[i,j]-std_mas_2[j]/C[j]**(0.5)*(1+zf[i]**2/2)**(0.5)*st.norm.ppf(0.95)
        nor_sup3[i,j]=normal3[i,j]+std_mas_3[j]/C[j]**(0.5)*(1+zf[i]**2/2)**(0.5)*st.norm.ppf(0.95)
        nor_inf3[i,j]=normal3[i,j]-std_mas_3[j]/C[j]**(0.5)*(1+zf[i]**2/2)**(0.5)*st.norm.ppf(0.95)
        #Intervalo de confianza Gumbel
        gum_sup1[i,j]=gumbel1[i,j]+ st.norm.ppf(0.95)*(std_mas_1[j]/C[j]**(0.5)*(1+1.1396*ktg[i,j]+1.1*ktg[i,j]**2)**(0.5))
        gum_inf1[i,j]=gumbel1[i,j]- st.norm.ppf(0.95)*(std_mas_1[j]/C[j]**(0.5)*(1+1.1396*ktg[i,j]+1.1*ktg[i,j]**2)**(0.5))
        gum_sup2[i,j]=gumbel2[i,j]+ st.norm.ppf(0.95)*(std_mas_2[j]/C[j]**(0.5)*(1+1.1396*ktg[i,j]+1.1*ktg[i,j]**2)**(0.5))
        gum_inf2[i,j]=gumbel2[i,j]- st.norm.ppf(0.95)*(std_mas_2[j]/C[j]**(0.5)*(1+1.1396*ktg[i,j]+1.1*ktg[i,j]**2)**(0.5))
        gum_sup3[i,j]=gumbel3[i,j]+ st.norm.ppf(0.95)*(std_mas_3[j]/C[j]**(0.5)*(1+1.1396*ktg[i,j]+1.1*ktg[i,j]**2)**(0.5))
        gum_inf3[i,j]=gumbel3[i,j]- st.norm.ppf(0.95)*(std_mas_3[j]/C[j]**(0.5)*(1+1.1396*ktg[i,j]+1.1*ktg[i,j]**2)**(0.5))
        #Intervalo de confianza Lognormal
        logn_sup1[i,j]=np.log(lognormal1[i,j])+std_mas_1[j]/C[j]**(0.5)*(1+zf[i]**2/2)**(0.5)*st.norm.ppf(0.95)
        logn_inf1[i,j]=np.log(lognormal1[i,j])-std_mas_1[j]/C[j]**(0.5)*(1+zf[i]**2/2)**(0.5)*st.norm.ppf(0.95)
        #Intervalo de confianza Pearson
        a=1-st.norm.ppf(0.95)**2/(2*(C[j]-1))
        b=ktp1[i,j]**2-st.norm.ppf(0.95)**2/C[j]
        kta[i,j]=(ktp1[i,j]+(ktp1[i,j]**2-a*b)**(0.5))/a
        ktb[i,j]=(ktp1[i,j]-(ktp1[i,j]**2-a*b)**(0.5))/a
        pear_sup1[i,j]=mean_mas_1[j]+kta[i,j]*std_mas_1[j]
        pear_inf1[i,j]=mean_mas_1[j]+ktb[i,j]*std_mas_1[j]
        pear_sup2[i,j]=mean_mas_2[j]+kta[i,j]*std_mas_2[j]
        pear_inf2[i,j]=mean_mas_2[j]+ktb[i,j]*std_mas_2[j]
        pear_sup3[i,j]=mean_mas_3[j]+kta[i,j]*std_mas_3[j]
        pear_inf3[i,j]=mean_mas_3[j]+ktb[i,j]*std_mas_3[j]

"""
np.savetxt('PDFnormal1D',normal1, delimiter=';')    
np.savetxt('PDFnormal2D',normal2,delimiter=';')  
np.savetxt('PDFnormal3D',normal3,delimiter=';') 
np.savetxt('PDFlognormal1D',lognormal1,delimiter=';')    
np.savetxt('PDFlognormal2D',lognormal2,delimiter=';')  
np.savetxt('PDFlognormal3D',lognormal3,delimiter=';')
np.savetxt('Gumbel1D',gumbel1,delimiter=';')  
np.savetxt('Gumbel2D',gumbel2,delimiter=';')  
np.savetxt('Gumbel3D',gumbel3,delimiter=';')
np.savetxt('Person1D',pearson1,delimiter=';')  
np.savetxt('Person2D',pearson2,delimiter=';')  
np.savetxt('Person3D',pearson3d,delimiter=';')"""
        #%% Graficas
a=0
plt.figure(1)
fig, axs = plt.subplots(3, 3)  
plt.title(str(1)+'d lluvia')
for j in range(0,9):
    if (normal[0,j]>0):
        r=j%3
        if (r==0): 
            a=a+1
        axs[a-1, r].plot(tr[:,j],mas_1s[j],'p',markersize=3,color='red')
        axs[a-1, r].plot(trf[:],normal1[:,j],'-',markersize=3,color='blue')
        #axs[a-1, r].plot(trf[:],lognormal1[:,j],'-.',markersize=3,color='orange')
        axs[a-1, r].plot(trf[:],gumbel1[:,j],'-',markersize=3,color='green')
        axs[a-1, r].plot(trf[:],pearson1[:,j],'-',markersize=3)
        axs[a-1,r].set_title('ZC'+str(j+1),fontsize=8)
        axs[a-1,r].grid(True,which="both", ls="-",color='gray')
        plt.tight_layout(pad=1, w_pad=1, h_pad=1.0)
        plt.legend(['mediciones','Normal','Lognormal', 'Gumbel','Pearson T3'],fontsize=6,ncol=1,loc=2)
        for ax in axs.flat:
            ax.set_xlabel('Tr (años)', fontsize=10)
            ax.set_ylabel('Pp (mm)', fontsize=10)
            ax.set_xscale('log')
            ax.set_xlim(0,1000)
            ax.set_ylim(0,200)
            ax.grid(True,which="both", ls="-",color='gray')
        for ax in axs.flat:
            ax.label_outer()
a=0
plt.figure(2)
fig, axs = plt.subplots(3, 3)
for j in range(9,18):
    if (normal[0,j]>0):
        r=j%3
        if (r==0): 
            a=a+1
        axs[a-1, r].plot(tr[:,j],mas_1s[j],'p',markersize=3, color='red')
        axs[a-1, r].plot(trf[:],normal1[:,j],'-',markersize=3,color='blue')
        #axs[a-1, r].plot(trf[:],lognormal1[:,j],'-.',markersize=3,color='orange')
        axs[a-1, r].plot(trf[:],gumbel1[:,j],'-',markersize=3,color='green')
        axs[a-1, r].plot(trf[:],pearson1[:,j],'-',markersize=3)
        axs[a-1,r].set_title('ZC'+str(j+1),fontsize=8)
        plt.legend(['mediciones','Normal','Lognormal', 'Gumbel','Pearson T3'],fontsize=6,ncol=1,loc=2)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        for ax in axs.flat:
            ax.set_xlabel('Tr (años)', fontsize=10)
            ax.set_ylabel('Pp (mm)', fontsize=10)
            ax.set_xscale('log')
            ax.set_xlim(0,1000)
            ax.set_ylim(0,200)
            ax.grid(True,which="both", ls="-",color='gray')
        for ax in axs.flat:
            ax.label_outer()

            
plt.figure(3)
plt.plot(tr[:,0]*len(mas_2s[0])/C[0],mas_2s[0],'p',markersize=3, color='red')
plt.plot(trf[:],normal2[:,0],'-',markersize=3,color='blue')
plt.plot(trf[:],nor_sup2[:,0],'-.',markersize=3,color='blue')
plt.plot(trf[:],nor_inf2[:,0],'-.',markersize=3,color='blue')
plt.title('ZC'+str(1)+' Tormenta 2 días',fontsize=8)
plt.xlabel('Tr (años)', fontsize=10)
plt.ylabel('Pp (mm)', fontsize=10)
plt.xscale('log')
plt.xlim(0,1000)
plt.ylim(0,150)
plt.grid(True,which="both", ls="-",color='gray')
plt.legend(['Mediciones','Normal','Banda de confianza'],fontsize=8,ncol=1,loc=2)
fig.savefig('ZC'+str(1)+'Tormenta 2 días -Normal')



plt.figure(3)
plt.plot(tr[:,0]*len(mas_2s[0])/C[0],mas_2s[0],'p',markersize=3, color='red')
#axs[a-1, r].plot(trf[:],lognormal1[:,j],'-.',markersize=3,color='orange')
plt.plot(trf[:],gumbel2[:,0],'-',markersize=3,color='blue')
plt.plot(trf[:],gum_sup2[:,0],'-.',markersize=3,color='blue')
plt.plot(trf[:],gum_inf2[:,0],'-.',markersize=3,color='blue')
plt.title('ZC'+str(1)+' Tormenta 2 días',fontsize=8)
plt.xlabel('Tr (años)', fontsize=10)
plt.ylabel('Pp (mm)', fontsize=10)
plt.xscale('log')
plt.xlim(0,1000)
plt.ylim(0,150)
plt.grid(True,which="both", ls="-",color='gray')
plt.legend(['Mediciones','Gumbel','Banda de Confianza'],fontsize=8,ncol=1,loc=2)
fig.savefig('ZC'+str(1)+'Tormenta 2 días -Gumbel')

plt.figure(3)
plt.plot(tr[:,0]*len(mas_2s[0])/C[0],mas_2s[0],'p',markersize=3, color='red')
plt.plot(trf[:],pearson2[:,0],'-',markersize=3, color='blue')
plt.plot(trf[:],pear_sup2[:,0],'-.',markersize=3,color='blue')
plt.plot(trf[:],pear_inf2[:,0],'-.',markersize=3,color='blue')
plt.title('ZC'+str(1)+' Tormenta 2 días',fontsize=8)
plt.xlabel('Tr (años)', fontsize=10)
plt.ylabel('Pp (mm)', fontsize=10)
plt.xscale('log')
plt.xlim(0,1000)
plt.ylim(0,150)
plt.grid(True,which="both", ls="-",color='gray')
plt.legend(['Mediciones','Pearson T3','Banda de Confianza'],fontsize=8,ncol=1,loc=2)
fig.savefig('ZC'+str(1)+'Tormenta 2 días -Pearson')
