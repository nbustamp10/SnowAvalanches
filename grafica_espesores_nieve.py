# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:53:36 2020

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


#date=[datetime.datetime(1980,1,1)+datetime.timedelta(days=x) for x in range (0,13948)]
#df=pd.DataFrame()
#df['date']=pd.to_datetime(date)
#df=df.set_index('date')
#df['year'] =pd.DatetimeIndex(df['date']).year
#df=df.set_index('year')

#isots=pd.read_csv('isotermas.csv',sep=';')
e=pd.read_csv('espesores_resumen.csv',sep=';')
e=np.array(e)

#%%
plt.figure(1)
for j in range(1,8):
   plt.plot(e[:,0],e[:,j]/1000,'p',markersize=7)
   plt.tight_layout(pad=3, w_pad=3, h_pad=0.5)
   plt.title('100',fontsize=14)
   #plt.plot(x_val,y_val,'-',color='black')
   #plt.plot(x_val1,y_val1,'-.',color='black')  
   plt.xlabel('Tr (años)', fontsize=14)
   plt.legend(['ZC1' ,'ZC2','ZC3','ZC4','ZC5','ZC6','ZC7'],fontsize=10,ncol=2,loc=2)
   plt.ylabel('e (m)', fontsize=14)
   plt.xscale('log')
   plt.xlim(0,1100)
   plt.ylim(0,2.3)
   plt.grid(True,which="both", ls="-",color='gray')
   
plt.figure(2)
for j in range(8,15):
   plt.plot(e[:,0],e[:,j]/1000,'p',markersize=7)
   plt.tight_layout(pad=3, w_pad=3, h_pad=0.5)
   plt.title('200',fontsize=14)
   #plt.plot(x_val,y_val,'-',color='black')
   #plt.plot(x_val1,y_val1,'-.',color='black')  
   plt.xlabel('Tr (años)', fontsize=14)
   plt.legend(['ZC1','ZC2','ZC3','ZC4','ZC5','ZC6','ZC7'],fontsize=10,ncol=2,loc=2)
   plt.ylabel('e (m)', fontsize=14)
   plt.xscale('log')
   plt.xlim(0,1100)
   plt.ylim(0,1.3)
   plt.grid(True,which="both", ls="-",color='gray')

plt.figure(9)
for j in range(15,22):
   plt.plot(e[:,0],e[:,j]/1000,'p',markersize=7)
   plt.tight_layout(pad=3, w_pad=3, h_pad=0.5)
   plt.title('400',fontsize=14)
   #plt.plot(x_val,y_val,'-',color='black')
   #plt.plot(x_val1,y_val1,'-.',color='black')  
   plt.xlabel('Tr (años)', fontsize=14)
   plt.legend(['ZC1','ZC2','ZC3','ZC4','ZC5','ZC6','ZC7'],fontsize=10,ncol=2,loc=2)
   plt.ylabel('e (m)', fontsize=14)
   plt.xscale('log')
   plt.xlim(0,1100)
   plt.ylim(0,.6)
   plt.grid(True,which="both", ls="-",color='gray')