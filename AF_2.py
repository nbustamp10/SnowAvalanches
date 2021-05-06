# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 00:02:34 2020

@author: User
"""

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
pp=pd.read_csv('Pp_pmgv2_zc1_v4.csv',sep=';')
temp=pd.read_csv('Temp_pmg_zc1_v4.csv',sep=';')
pp=np.array(pp)
temp=np.array(temp)
temp=temp[:,1:]
a=len(pp)
S=(a,7)
S2=(39,7)
S1=(a,6)
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
    for j in range(1,8):
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
t_tormenta1=pd.DataFrame(tormentas1[0:,0:], columns=['ZC1', 'ZC2', 'ZC3','ZC4','ZC5','ZC6','ZC7'],index=df.index)
t_tormenta2=pd.DataFrame(tormentas2[0:,0:], columns=['ZC1', 'ZC2', 'ZC3','ZC4','ZC5','ZC6','ZC7'],index=df.index)
t_tormenta3=pd.DataFrame(tormentas3[0:,0:], columns=['ZC1', 'ZC2', 'ZC3','ZC4','ZC5','ZC6','ZC7'],index=df.index)
t_tormenta4=pd.DataFrame(tormentas4[0:,0:], columns=['ZC1', 'ZC2', 'ZC3','ZC4','ZC5','ZC6','ZC7'],index=df.index)
t_tormenta5=pd.DataFrame(tormentas5[0:,0:], columns=['ZC1', 'ZC2', 'ZC3','ZC4','ZC5','ZC6','ZC7'],index=df.index)

for k in range(0,len(pp)-4):
    for j in range(0,7):
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
t_snow1=pd.DataFrame(snow1[0:,0:], columns=['ZC1', 'ZC2', 'ZC3','ZC4','ZC5','ZC6','ZC7'],index=df.index)
t_snow2=pd.DataFrame(snow2[0:,0:], columns=['ZC1', 'ZC2', 'ZC3','ZC4','ZC5','ZC6','ZC7'],index=df.index)
t_snow3=pd.DataFrame(snow3[0:,0:], columns=['ZC1', 'ZC2', 'ZC3','ZC4','ZC5','ZC6','ZC7'],index=df.index)
t_snow4=pd.DataFrame(snow4[0:,0:], columns=['ZC1', 'ZC2', 'ZC3','ZC4','ZC5','ZC6','ZC7'],index=df.index)
t_snow5=pd.DataFrame(snow5[0:,0:], columns=['ZC1', 'ZC2', 'ZC3','ZC4','ZC5','ZC6','ZC7'],index=df.index)


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
mas_1s.to_csv('MaxA-nieve-1d', sep='&',float_format='%.1f')
mas_2s.to_csv('MaxA-nieve-2d', sep='&',float_format='%.1f')
mas_3s.to_csv('MaxA-nieve-3d', sep='&',float_format='%.1f')
mas_4s.to_csv('MaxA-nieve-4d', sep='&',float_format='%.1f')
mas_5s.to_csv('MaxA-nieve-5d', sep='&',float_format='%.1f')
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
b=np.zeros(len(mean_mas_1))
C=np.zeros(len(mean_mas_1))
se=np.zeros(len(mean_mas_1))
cs=np.zeros(len(mean_mas_1))
yn1=np.zeros(len(mean_mas_1))
sigman1=np.zeros(len(mean_mas_1))
p=np.zeros(S2)
z=np.zeros(S2)
ktd=np.zeros(S2)
ktp=np.zeros(S2)
ktlp=np.zeros(S2)
chisq=np.zeros((5,len(mean_mas_1)))
ch2=np.zeros((5,len(mean_mas_1)))

fn=mas_1s
mean=mean_mas_1
std=std_mas_1
logmean=mean_log_mas1
logstd=std_log_mas1

clasew=np.zeros(10)
auxiliarp=pd.DataFrame()
for j in range(0,7):#len(mean_mas_1)):
    b[j]=np.count_nonzero(np.isnan(fn[j]))
    normalp=pd.DataFrame()
    lognormalp=pd.DataFrame()
    gumbelp=pd.DataFrame()
    pearsonp=pd.DataFrame()
    logpearsonp=pd.DataFrame()
    for k in range(0,a):
        auxiliar=fn[j].values
        c=np.argwhere(np.isnan(auxiliar)).min()
        C[j]=c
        auxiliarp=pd.DataFrame(auxiliar[:c],columns=['pp'])
        auxiliarp['int'] = pd.cut(auxiliarp['pp'], bins=10)
        clasew=auxiliarp['int'].value_counts()
        np.savetxt('int_w'+str(j),clasew,delimiter='&',fmt='%.2f') 
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
            cs[j]=(c/((c-1)*(c-2)))*(((np.log(auxiliar[:c])-logmean[j])/logstd[j])**3).sum()
            ktp[k,j]=pearson3.ppf([1-p[k,j]], se[j]) #Factor de frecuencia de Pearson3
            z[k,j]=st.norm.ppf(1-p[k,j]) #K de variable normal estandalizada
            ktd[k,j]=(-np.log(np.log(tr[k,j]/(tr[k,j]-1)))-yn)/sigman #Kt-Gumbel
            ktlp[k,j]=z[k,j]+(z[k,j]**2-1)*(cs[j]/6)+1/3*(z[k,j]**3-6*z[k,j])*((cs[j]/6)**2)-(z[k,j]**2-1)*(cs[j]/6)**3+z[k,j]*(cs[j]/6)**4+1/3*(cs[j]/6)**5
            gumbel[k,j]=mean[j]+ktd[k,j]*std[j]
            normal[k,j]=mean[j]+z[k,j]*std[j]
            lognormal[k,j]=np.exp(logmean[j]+logstd[j]*z[k,j])
            pearson[k,j]=mean[j]+ktp[k,j]*std[j]
            logpearson[k,j]=np.exp(logmean[j]+logstd[j]*ktlp[k,j])            
"""            
            normalc = pd.DataFrame(normal[:c, j],columns=['data'])
            gumbelc = pd.DataFrame(gumbel[:c, j],columns=['data'])
            pearsonc = pd.DataFrame(pearson[:c, j],columns=['data'])
            logpearsonc=pd.DataFrame(logpearson[:c,j],columns=['data'])
            
            normalp['int'] = pd.cut(normalc['data'], bins=10)
            clasen=normalp['int'].value_counts()
            lognormalp['int'] = pd.cut(normalc['data'], bins=10)
            claseln=lognormalp['int'].value_counts()
            gumbelp['int'] = pd.cut(gumbelc['data'], bins=10)
            claseg=gumbelp['int'].value_counts()
            pearsonp['int'] = pd.cut(pearsonc['data'], bins=10)
            clasep=pearsonp['int'].value_counts()
            logpearsonp['int'] = pd.cut(pearsonc['data'], bins=10)
            claselp=logpearsonp['int'].value_counts()
            np.savetxt('int_normal'+str(j),clasen,delimiter='&',fmt='%.2f')  
            np.savetxt('int_gumbel'+str(j),claseg,delimiter='&',fmt='%.2f') 
            np.savetxt('int_pearson'+str(j),clasep,delimiter='&',fmt='%.2f') 
            np.savetxt('int_lognormal'+str(j),claseln,delimiter='&',fmt='%.2f') 
            np.savetxt('int_logpearson'+str(j),claselp,delimiter='&',fmt='%.2f') 
  #%%                  
            'Bondad de ajutste'
            chisq[0,j]=chisquare(np.clasew,np.clasen)[0]       #Bondad de ajuste para la normal
            chisq[1,j]=chisquare(clasew[:c],claseln[:c,j])[0]    #Bondad de ajuste para la lognormal
            chisq[2,j]=chisquare(clasew[:c],claseg[:c,j])[0]       #Bondad de ajuste para la Gumbel
            chisq[3,j]=chisquare(clasew[:c],clasep[:c,j])[0]      #Bondad de ajuste para la Pearson
            chisq[4,j]=chisquare(clasew[:c],claselp[:c,j])[0]   #Bondad de ajuste para la logpearson

"""
        #for i in range(0,5):
         #   if (chisq[i,j]<=0.05):
          #      ch2[i,j]=0 #rechazo
           # else:2
            #    ch2[i,j]=1 #acepta
#%%        
'Analisis para periodos posteriores'
tf=[1.5,2,5,10,25,30,50,100,200,1000]

S1=(len(tf),len(mean_mas_1))
normalt=np.zeros(S1)
gumbelt=np.zeros(S1)
lognormalt=np.zeros(S1)
pearsont=np.zeros(S1)
logpearsont=np.zeros(S1)
nor_sup=np.zeros(S1)
nor_inf=np.zeros(S1)
gum_sup=np.zeros(S1)
gum_inf=np.zeros(S1)
pear_sup=np.zeros(S1)
pear_inf=np.zeros(S1)
logn_sup=np.zeros(S1)
logn_inf=np.zeros(S1)
en=np.zeros(S1)
ena=np.zeros((len(tf),len(mean_mas_1),4))
riesgo=np.zeros(S1)
kta=np.zeros(S1)
ktb=np.zeros(S1)

w1=np.zeros(len(tf))

trf=np.zeros((len(tf),len(mean)))
ktg=np.zeros(S1)
ktp1=np.zeros(S1)
ktlp1=np.zeros(S1)
rhos=[50,100,200,400]
for j in range(0,len(mean_mas_2)):
    
    for i in range(0,len(tf)):
        trf[i,j]=tf[i]*C[j]/len(fn[j])
        if (C[j]>=20):
            t=trf[i,j]
            pf[i]=1/t
            zf[i]=st.norm.ppf(1-pf[i])
            #ktg[i]=gumbel_r.ppf(1-pf[i])
            ktg[i,j]=(-np.log(np.log(t/(t-1)))-yn1[j])/sigman1[j]
            ktp1[i,j]=pearson3.ppf([1-pf[i]], se[j])
            ktlp1[i,j]=zf[i]+(zf[i]**2-1)*(cs[j]/6)+1/3*(zf[i]**3-6*zf[i])*((cs[j]/6)**2)-(zf[i]**2-1)*(cs[j]/6)**3+zf[i]*(cs[j]/6)**4+1/3*(cs[j]/6)**5
            normalt[i,j]=mean[j]+zf[i]*std[j]
            gumbelt[i,j]=mean[j]+ktg[i,j]*std[j]
            lognormalt[i,j]=np.exp(logmean[j]+logstd[j]*zf[i])
            pearsont[i,j]=mean[j]+ktp1[i,j]*std[j]
            logpearsont[i,j]=np.exp(logmean[j]+logstd[j]*ktlp1[i,j])
            #intervalos de confianza Normal
            nor_sup[i,j]=normalt[i,j]+std[j]/C[j]**(0.5)*(1+zf[i]**2/2)**(0.5)*st.norm.ppf(0.95)
            nor_inf[i,j]=normalt[i,j]-std[j]/C[j]**(0.5)*(1+zf[i]**2/2)**(0.5)*st.norm.ppf(0.95)
            #Intervalo de confianza Gumbel
            gum_sup[i,j]=gumbelt[i,j]+ st.norm.ppf(0.95)*(std[j]/C[j]**(0.5)*(1+1.1396*ktg[i,j]+1.1*ktg[i,j]**2)**(0.5))
            gum_inf[i,j]=gumbelt[i,j]- st.norm.ppf(0.95)*(std[j]/C[j]**(0.5)*(1+1.1396*ktg[i,j]+1.1*ktg[i,j]**2)**(0.5))
            #Intervalo de confianza Lognormal
            logn_sup[i,j]=np.exp(np.log(lognormalt[i,j])+logstd[j]/C[j]**(0.5)*(1+zf[i]**2/2)**(0.5)*st.norm.ppf(0.95))
            logn_inf[i,j]=np.exp(np.log(lognormalt[i,j])-logstd[j]/C[j]**(0.5)*(1+zf[i]**2/2)**(0.5)*st.norm.ppf(0.95))
            #Intervalo de confianza Pearson
            a=1-st.norm.ppf(0.95)**2/(2*(C[j]-1))
            b=ktp1[i,j]**2-st.norm.ppf(0.95)**2/C[j]
            kta[i,j]=(ktp1[i,j]+(ktp1[i,j]**2-a*b)**(0.5))/a
            ktb[i,j]=(ktp1[i,j]-(ktp1[i,j]**2-a*b)**(0.5))/a
            pear_sup[i,j]=mean[j]+kta[i,j]*std[j]
            pear_inf[i,j]=mean[j]+ktb[i,j]*std[j]

for k in range(0,len(rhos)):
    for j in range(0,len(mean_mas_2)):
        for i in range(0,len(tf)):
            f=rhos[k]/1000
            en[i,j]=pearsont[i,j]/f #Espesores de nieve
            ena[i,j,k]=pearsont[i,j]/f
    name='Espesorp1'+str(rhos[k])
    np.savetxt(name, en, delimiter='&', fmt='%.0f')


p1=[30,0]
p2=[30,1000]
x_val=[p1[0], p2[0]]
y_val=[p1[1],p2[1]]

p3=[300,0]
p4=[300,1000]
x_val1=[p3[0], p4[0]]
y_val1=[p3[1],p4[1]]

fig, axs1= plt.subplots(7, 1,figsize=(10, 10), sharey=True) 
#plt.suptitle('Espesores de nieve (mm)',fontsize=12)
for k in range(1,4):
     for j in range(0,7):
        axs1[j].plot(tf[2:],ena[2:,j,k]/1000,'p',markersize=7)
        plt.tight_layout(pad=3, w_pad=3, h_pad=0.5)
        axs1[j].set_title('ZC'+str(j+1),fontsize=14)
        axs1[j].plot(x_val,y_val,'-',color='black')
        axs1[j].plot(x_val1,y_val1,'-.',color='black')  
        for ax in axs1.flat:
            ax.set_xlabel('Tr (años)', fontsize=14)
            ax.legend(['100','Tr30', 'Tr300','200', '400'],fontsize=14,ncol=5,loc=2)
            ax.set_ylabel('e (m)', fontsize=14)
            ax.set_xscale('log')
            ax.set_xlim(0,1100)
            ax.set_ylim(0,1.5)
            ax.grid(True,which="both", ls="-",color='gray')
        for ax in axs1.flat:
            ax.label_outer()

p1=[30,0]
p2=[30,1000]
x_val=[p1[0], p2[0]]
y_val=[p1[1],p2[1]]

p3=[300,0]
p4=[300,1000]
x_val1=[p3[0], p4[0]]
y_val1=[p3[1],p4[1]]

#fig, axs1= plt.subplots(3, 1,figsize=(10, 10), sharey=True) 
#plt.suptitle('Espesores de nieve (mm)',fontsize=12)
plt.figure(7)
for j in range(1,7):
   plt.plot(tf[2:],ena[2:,j,1]/1000,'p',markersize=7)
   plt.tight_layout(pad=3, w_pad=3, h_pad=0.5)
   plt.title('100',fontsize=14)
   #plt.plot(x_val,y_val,'-',color='black')
   #plt.plot(x_val1,y_val1,'-.',color='black')  
   plt.xlabel('Tr (años)', fontsize=14)
   plt.legend(['ZC2','ZC3','ZC4','ZC5','ZC6','ZC7'],fontsize=10,ncol=2,loc=4)
   plt.ylabel('e (m)', fontsize=14)
   plt.xscale('log')
   plt.xlim(0,1100)
   plt.ylim(0,2)
   plt.grid(True,which="both", ls="-",color='gray')
   
plt.figure(8)
for j in range(1,7):
   plt.plot(tf[2:],ena[2:,j,2]/1000,'p',markersize=7)
   plt.tight_layout(pad=3, w_pad=3, h_pad=0.5)
   plt.title('200',fontsize=14)
   #plt.plot(x_val,y_val,'-',color='black')
   #plt.plot(x_val1,y_val1,'-.',color='black')  
   plt.xlabel('Tr (años)', fontsize=14)
   plt.legend(['ZC2','ZC3','ZC4','ZC5','ZC6','ZC7'],fontsize=10,ncol=2,loc=4)
   plt.ylabel('e (m)', fontsize=14)
   plt.xscale('log')
   plt.xlim(0,1100)
   plt.ylim(0,1)
   plt.grid(True,which="both", ls="-",color='gray')

plt.figure(9)
for j in range(1,7):
   plt.plot(tf[2:],ena[2:,j,3]/1000,'p',markersize=7)
   plt.tight_layout(pad=3, w_pad=3, h_pad=0.5)
   plt.title('400',fontsize=14)
   #plt.plot(x_val,y_val,'-',color='black')
   #plt.plot(x_val1,y_val1,'-.',color='black')  
   plt.xlabel('Tr (años)', fontsize=14)
   plt.legend(['ZC2','ZC3','ZC4','ZC5','ZC6','ZC7'],fontsize=10,ncol=2,loc=4)
   plt.ylabel('e (m)', fontsize=14)
   plt.xscale('log')
   plt.xlim(0,1100)
   plt.ylim(0,.6)
   plt.grid(True,which="both", ls="-",color='gray')




#np.savetxt('chisq_weibull1d',chisq,delimiter='&',fmt='%.2f')  



       
"""    
np.savetxt('Gumbel3D_weibull', gumbel, delimiter=';', fmt='%.2f')
np.savetxt('Normal3D_weibull', normal, delimiter=';')
np.savetxt('Lognormal3D_weibull', lognormal,delimiter=';',fmt='%.2f) 
np.savetxt('Pearson3d_weibull', pearson,delimiter=';',fmt='%.2f) 
np.savetxt('Tr_weibull3d', tr,delimiter=';',fmt='%.2f) 
 
np.savetxt('PDFnormal3D',normalt, delimiter=';',fmt='%.2f)    
np.savetxt('PDFlognormal3D',lognormalt,delimiter=';',fmt='%.2f)    
np.savetxt('Gumbel3D',gumbelt,delimiter=';',fmt='%.2f)  
np.savetxt('Person3D',pearsont,delimiter=';',fmt='%.2f)"""

"Presión de impacto" 
m=0
dz=np.zeros(7) + (211, 362, 333, 92, 48, 246, 118)        
v=1.8*dz**(0.5)      
ip=np.zeros((4,len(dz)))
for i in rhos: 
    for j in range(0, len(dz)):
        ip[m,j]=i*v[j]**2/1000
    m=m+1

for k in range(0,len(rhos)):  
    for j in range(0,len(mean_mas_2)):
       for i in range(0,len(tf)): 
               if (ena[i,j,k]>250 and ip[k,j]>30): 
                   riesgo[i,j]=250
               if (ena[i,j,k]>500  and ip[k,j]>30) :
                   riesgo[i,j]=500
    np.savetxt('Riesgo2d'+str(rhos[k]),riesgo,delimiter=';',fmt='%.0f')
        #%% Graficas
plt.figure(1)
plt.plot(tr[:,6]*len(fn[6])/C[6],fn[6],'p',markersize=3, color='red')
plt.plot(tf[:],normalt[:,6],'-',markersize=3,color='blue')
plt.plot(tf[:],nor_sup[:,6],'-.',markersize=3,color='blue')
plt.plot(tf[:],nor_inf[:,6],'-.',markersize=3,color='blue')
plt.title('ZC'+str(7)+' Tormenta 1 día',fontsize=14)
plt.xlabel('Tr (años)', fontsize=14)
plt.ylabel('Pp (mm)', fontsize=14)
plt.xscale('log')
plt.xlim(0,1000)
plt.ylim(0,200)
plt.grid(True,which="both", ls="-",color='gray')
plt.legend(['Mediciones','Normal','Banda de confianza'],fontsize=8,ncol=1,loc=2)
#plt.savefig('ZC'+str(7)+'Tormenta 1 día -Normal')



plt.figure(2)
plt.plot(tr[:,6]*len(fn[6])/C[6],fn[6],'p',markersize=3, color='red')
#axs[a-1, r].plot(trf[:],lognormal1[:,j],'-.',markersize=3,color='orange')
plt.plot(tf[:],gumbelt[:,6],'-',markersize=3,color='blue')
plt.plot(tf[:],gum_sup[:,6],'-.',markersize=3,color='blue')
plt.plot(tf[:],gum_inf[:,6],'-.',markersize=3,color='blue')
plt.title('ZC'+str(7)+' Tormenta 1 día',fontsize=14)
plt.xlabel('Tr (años)', fontsize=14)
plt.ylabel('Pp (mm)', fontsize=14)
plt.xscale('log')
plt.xlim(0,1000)
plt.ylim(0,200)
plt.grid(True,which="both", ls="-",color='gray')
plt.legend(['Mediciones','Gumbel','Banda de Confianza'],fontsize=8,ncol=1,loc=2)
#plt.savefig('ZC'+str(7)+'Tormenta 1 día -Gumbel')

plt.figure(3)
plt.plot(tr[:,6]*len(fn[6])/C[6],fn[6],'p',markersize=3, color='red')
plt.plot(tf[:],pearsont[:,6],'-',markersize=3, color='blue')
plt.plot(tf[:],pear_sup[:,6],'-.',markersize=3,color='blue')
plt.plot(tf[:],pear_inf[:,6],'-.',markersize=3,color='blue')
plt.title('ZC'+str(7)+' Tormenta 1 día',fontsize=14)
plt.xlabel('Tr (años)', fontsize=14)
plt.ylabel('Pp (mm)', fontsize=14)
plt.xscale('log')
plt.xlim(0,1000)
plt.ylim(0,200)
plt.grid(True,which="both", ls="-",color='gray')
plt.legend(['Mediciones','Pearson T3','Banda de Confianza'],fontsize=8,ncol=1,loc=2)
#plt.savefig('ZC'+str(7)+'Tormenta 1 día -Pearson')

plt.figure(4)
plt.loglog(tr[:,6]*len(fn[6])/C[6],fn[6],'p',markersize=3, color='red')
plt.loglog(tf[:],lognormalt[:,6],'-',markersize=3, color='blue')
plt.loglog(tf[:],logn_sup[:,6],'-.',markersize=3,color='blue')
plt.loglog(tf[:],logn_inf[:,6],'-.',markersize=3,color='blue')
plt.title('ZC'+str(7)+' Tormenta 1 día',fontsize=14)
plt.xlabel('Tr (años)', fontsize=14)
plt.ylabel('Pp (mm)', fontsize=14)
#plt.xscale('log')
plt.xlim(0,1000)
plt.ylim(0,1000)
plt.grid(True,which="both", ls="-",color='gray')
plt.legend(['Mediciones','Lognormal','Banda de Confianza'],fontsize=8,ncol=1,loc=2)
#plt.savefig('ZC'+str(1)+'Tormenta 1 día -Lognormal')

a=0
plt.figure(5)
fig, axs = plt.subplots(2, 4)  
plt.title(str(1)+'d lluvia',fontsize=14)
for j in range(0,7):
    if (normal[0,j]>0):
        r=j%4
        if (r==0): 
            a=a+1
        axs[a-1, r].plot(tr[:,j],fn[j],'p',markersize=3,color='red')
        axs[a-1, r].plot(tf[:],normalt[:,j],'-',markersize=3,color='blue')
        #axs[a-1, r].plot(trf[:],lognormal1[:,j],'-.',markersize=3,color='orange')
        axs[a-1, r].plot(tf[:],gumbelt[:,j],'-',markersize=3,color='green')
        axs[a-1, r].plot(tf[:],pearsont[:,j],'-',markersize=3)
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


