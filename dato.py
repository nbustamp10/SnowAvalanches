import pandas as pd
from bs4 import BeautifulSoup as bs
import os
import requests  as rq
import urllib
import calendar

#import string    as str
z=0
marge=pd.DataFrame()
missing_dates = []
# Leer <PRE>
for i in range(2000,2019):
   for j in range(1,13):
       a=calendar.monthrange(i, j)[1]
       for d in range(1,a+1):
          inicio='http://weather.uwyo.edu/cgi-bin/sounding?region=naconf&TYPE=TEXT%3ALIST&YEAR='
          fin='&TO=2912&STNM=85586'
          for m in ["00", "12"]:
              url=inicio + str(i) + '&MONTH=' + str(j) + '&dialy&FROM='+str(d)+m+fin
              r = rq.get(url)
              soup = bs(r.content, 'lxml')
              try:
                  pre = soup.select_one('pre').text
                  results = []
                  for line in pre.split('\n')[1:-1]:
                      if '--' not in line:
                          row = [line[n:n+7].strip() for n in range(0, len(line), 7)]
                          results.append(row)
                  df = pd.DataFrame(results)
                  
                  f='{}{}{}{}{}{}{}{}{}'.format('GS','-',i,'-',str(j).zfill(2),'-',str(d).zfill(2),'-',m)+'.csv'
                  df.to_csv(f,sep=';')
                  print(df) 
              except:
                  missing_dates.append([i,j,d])
                  continue
    
    
       # df=pd.DataFrame(results)
#     f='{}{}{}{}{}'.format(z,'-',i,'-',j)+'.csv'
#     Datos.to_csv(f,sep=';')
     #print (df)
     #marge=marge.append(Datos)
     
# Leer tabla     
#for i in range(2000,2021):
 #   for j in range(1,10):
  #   inicio='http://weather.uwyo.edu/cgi-bin/sounding?region=naconf&TYPE=TEXT%3ALIST&YEAR='
   #  fin='&FROM=1112&TO1112&STNM=85586'
    # url=inicio + str(i) + '&MONTH=0' + str(j) + fin
     #r = rq.get(url)
     #Datos = pd.read_html(r.content)[1]
     #z+=1
     #f='{}{}{}{}{}'.format(z,'-',i,'-',j)+'.csv'
     #Datos.to_csv(f,sep=';')
     #print (z)
     #marge=marge.append(Datos)

     #print(sfiles)
#marge.to_csv('Temp_min.csv',sep=';')