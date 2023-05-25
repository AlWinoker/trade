# -*- coding: utf-8 -*-
"""
Created on Wed May 24 09:54:44 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 20:19:01 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 14:29:10 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 13:46:52 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 14:47:38 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:52:38 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 20:44:06 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 10:23:49 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:57:55 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 11:36:19 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 22:58:33 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 21:55:01 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 19:26:53 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 20:43:29 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 13:03:04 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 21:26:07 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 19:00:03 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 20:51:17 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 19:50:00 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 22:57:33 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 18:23:05 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 16:08:39 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:37:38 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:19:22 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 13:06:28 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 11:20:04 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 17:32:09 2023

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 14:10:45 2022

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 10:01:52 2022

@author: 14015
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 08:32:37 2022

@author: 14015
"""



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, RNN
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque

import os
import numpy as np
import pandas as pd
import random

import requests
import matplotlib.pyplot as plt
import numpy as np
import time 
from urllib.parse import urljoin, urlencode
import json
import hmac
import hashlib
import random
from datetime import datetime

from binance.client import Client

#tickerPairs = ['USDT','BUSDUSDT','ETHUSDT','BTCBUSD','BTCUSDC','USDCUSDT','ETHBUSD','MATICUSDT','ETHBTC','USDCBUSD','SOLUSDT','BURGERUSDT','AAVEUSDT','AVAXBUSD','BNBUSDT','RUNEBUSD','VIDTUSDT','ETHUSDC','LRCUSDT','MATICBUSD','WAVESUSDT','ADAUSDT','BURGERBUSD','SHIBBUSD','SOLBUSD','CHZUSDT','BTCGBP','BNBBUSD','UNIUSDT','BTCEUR','ETHEUR','UNFIUSDT','SNXUSDT','LUNCBUSD','DOTBUSD','MATICBTC','SHIBUSDT','XRPUSDT','SCRTUSDT','DYDXUSDT','SANDUSDT','LUNAUSDT','BAKEUSDT','PERPUSDT','JASMYUSDT','ONEUSDT','XMRUSDT','EGLDUSDT','FILUSDT','ELFUSDT','BTCSTUSDT','LUNABUSD','XTZUSDT','ATOMUSDT','TROYBUSD','SUSHIUSDT']
#tickerPairs = ['BTCBUSD','BTCUSDT','BUSDUSDT','ETHUSDT','VGXUSDT','SOLUSDT','BTCUSDC','ETHBUSD','BNBUSDT','ETHBTC','USDCUSDT','BURGERUSDT','MATICUSDT','ADAUSDT','ADAUSDT','BTCEUR','USDCBUSD','BNBBUSD','KMDUSDT','AVAXBUSD','GMTUSDT','USTCBUSD','SOLBUSD','WINUSDT','ADABUSD','BURGERBUSD','AAVEUSDT','UNIUSDT','VGXBTC','ETHUSDC','XRPUSDT','SHIBUSDT','USDTTRY','ETHEUR','FTMUSDT','LUNCBUSD','APEUSDT','TRXUSDT','MATICBUSD','SANDUSDT','BTCUSDT','DOTUSDT','GALAUSDT','LINKUSDT','RUNEUSDT','XRPBUSD','NEARUSDT','FILUSDT','WTCUSDT','JASMYUSDT','STPTBUSD','BUSDTRY','DYDXUSDT','OGNUSDT','WAVESUSDT','ANCUSDT','LDOUSDT','ICPUSDT','MANAUSDT','VIDTUSDT','RUNEBUSD','AVAXUSDT','BTCTRY','ATOMUSDT','BEAMUSDT','LTCUSDT','APEBUSD','BTCGBP','ETHGBP','CHZUSDT','REIUSDT','FLMUSDT','SANDBUSD','AVAXTRY','NEARBUSD','BTCSTUSDT','SHIBBUSD','DOTBUSD','CKBUSDT','UNFIUSDT','GMTBUSD','LUNAUSDT','EGLDUSDT','AUDUSDT','ETHTRY','VETUSDT','ZILUSDT','SRMUSDT','ONEUSDT','CRVUSDT','BNBBTC','LUNABUSD','BTCAUD','TROYBUSD','PEOPLEUSDT','ANCBUSD','USDTBRL','KEYBUSD','OPUSDT','GBPUSDT','WRXUSDT','QNTUSDT','OCEANUSDT','XTZUSDT','TUSDUSDT','BUSDBRL','LTCBUSD','ALGOUSDT','ARUSDT','BCHUSDT','ROSEUSDT','AAVEBUSD','USDTDAI','ADXUSDT','BTCSTBUSD','BTCBRL','ETCUSDT','SHIBTRY','AXSUSDT','MINAUSDT','RSRUSDT','COMPUSDT','LDOBUSD','BTTCUSDT','BTCBUSD','ATOMBUSD','ACHBUSD','SNXUSDT','AUDBUSD','ZECUSDT','BELUSDT','TLMUSDT','ICPBUSD','XRPBTC','XMRUSDT','GALABUSD','MBLUSDT','SCRTUSDT','TVKUSDT','PNTUSDT','ALICEUSDT','KMDBTC','SLPUSDT','BTCDOWNUSDT','LEVERUSDT','WAVESBTC','XMRBTC','REIBUSD','WBTCBTC','BELTRY','YFIIUSDT','RUNEBTC','RUNEBTC','WOOUSDT','SUNUSDT','CTXCUSDT','NULSUSDT','PAXGBUSD','STPTUSDT','HIGHUSDT','NMRUSDT','SOLUSDC','TRXBUSD','STORJUSDT','LINKBTC','GRTUSDT','BNBETH','ETHDAI','LINKBUSD','KEYUSDT','MATICBTC','ELFUSDT','XLMUSDT','PAXGUSDT','EOSUSDT','LRCUSDT','SOLBTC','ETHDOWNUSDT','FTMBUSD','BAKEUSDT','FTTUSDT','GALUSDT','VGXETH','WINBUSD','VIDTBUSD','PYRUSDT','EPXUSDT','ETHAUD','MOVRUSDT','JSTUSDT','JASMYBUSD','PERPUSDT','SUSHIUSDT','CKBBUSD','ENSUSDT','ADABTC','THETAUSDT','DARUSDT','AVAXBTC','ENJUSDT','UNIBUSD','LTCBTC','UNFIBUSD','BTCUPUSDT','ENJUSDT','MIRUSDT','SOLEUR','DCRUSDT','ETHUPUSDT','DOTBTC','CAKEUSDT','LITUSDT','YFIUSDT','HOTUSDT','DASHUSDT','IMXUSDT','IOTXUSDT','WAVESBUSD','VTHOUSDT','ROSEBUSD','GBPBUSD','FTTBUSD','USDTBIDR','EGLDBUSD','LINKUPUSDT','WTCBTC','FILBUSD','ADAUPUSDT','UMAUSDT','WRXBUSD','GLMRUSDT','OPBUSD','OPBUSD','COCOSTRY','CVXUSDT','UNIBTC','IOTAUSDT','SCRTBUSD','CELOUSDT','BIFIUSDT','SPELLUSDT','LOOMBTC','IOSTUSDT','IDEXUSDT','AXSBUSD','SOLTRY','VITEUSDT','ONEBUSD','DYDXBUSD','GMTTRY','MANABUSD','KLAYUSDT','TRXBTC','WANUSDT','ELFBUSD','BLZUSDT','KDABUSD','QTUMUSDT','CELRUSDT','MASKUSDT','SANDBTC','AAVEBTC','EPXBUSD','FIROUSDT','SPELLTRY','SXPUSDT','MFTUSDT','CHRUSDT','XRPUSDC','HBARUSDT','LITBUSD','BADGERUSDT','SSVBTC','TRBUSDT','BTCBTC','OXTUSDT','SANTOSTRY','USDTUAH','RVNUSDT','KAVAUSDT','DUSKUSDT','VETBUSD','WNXMUSDT','FTMTRY','BTCDAI','NBSUSDT','BETHETH','SKLUSDT','ADAUSDC','TLMTRY','1INCHUSDT','OMUSDT','QNTBUSD','ATOMBTC','GMTBTC','DODOUSDT','XRPEUR','ADADOWNUSDT','COTIUSDT','ARPATRY','RENUSDT','SFPUSDT','BELBUSD','KP3RUSDT','BNXUSDT','BNBEUR','TLMBUSD','ELFBTC','RLCUSDT','AUDIOUSDT','BEAMBTC','HNTUSDT','LEVERBUSD','AERGOBUSD','JASMYBTC','ONTUSDT','SANTOSUSDT','ANKRUSDT','ATAUSDT','ARPAUSDT','MKRUSDT','OGUSDT','CHZBUSD','UGNBUSD','RAMPUSDT','BCHBUSD','DOTDOWNUSDT','BUSDBIDR','DOTUPUSDT','LINKDOWNUSDT','KSMUSDT','CTSIUSDT','WAVESTRY','RUNEETH','SUPERUSDT','APEBTC','RADUSDT','STRAXUSDT','MULTIUSDT','TVKBUSD','ZILBUSD','ILVUSDT','PERPBUSD','QNTBTC','GTOUSDT','SRMBUSD','JOEUSDT','LINAUSDT','LINKETH','API3USDT','LOKAUSDT','OMGUSDT','BTCSTBTC','MANABTC','NMRBUSD','QIUSDT','ALGOBUSD','FETUSDT','BNBUSDC']
#tickerPairs = ['ZRXUSDT','1INCHUSDT','AAVEUSDT','GHSTUSDT','ACAUSDT','AGLDUSDT','ALCXUSDT','ACHUSDT','ALGOUSDT','TLMUSDT','ADXUSDT','FORTHUSDT','ANKRUSDT','APEUSDT','API3USDT','ANTUSDT','ASTRUSDT','AUDIOUSDT','REPUSDT','AVAXUSDT','AXSUSDT','BADGERUSDT','BNTUSDT','BALUSDT','BANDUSDT','BONDUSDT','BATUSDT','BICOUSDT','BTCUSDT','BCHUSDT','BTTUSDT','FIDAUSDT','ADAUSDT','CTSIUSDT','LINKUSDT','CHZUSDT','CHRUSDT','CVCUSDT','COMPUSDT','CVXUSDT','ATOMUSDT','COTIUSDT','CRVUSDT','DAIUSDT','DASHUSDT','MANAUSDT','DENTUSDT','BTCUSDT','DYDXUSDT','EGLDUSDT','ENJUSDT','MLNUSDT','EOSUSDT','ETHUSDT','ETCUSDT','ENSUSDT','FTMUSDT','FETUSDT','FILUSDT','FLOWUSDT','FXSUSDT','GALAUSDT','GTCUSDT','GNOUSDT','FARMUSDT','ICXUSDT','IDEXUSDT','RLCUSDT','IMXUSDT','INJUSDT','ICPUSDT','JASMYUSDT','KAVAUSDT','KEEPUSDT','KP3RUSDT','KSMUSDT','KNCUSDT','LDOUSDT','LSKUSDT','LTCUSDT','LPTUSDT','LRCUSDT','MCUSDT','MULTIUSDT','ALICEUSDT','MKRUSDT','MASKUSDT','MINAUSDT','MIRUSDT','XMRUSDT','GLMRUSDT','MOVRUSDT','NANOUSDT','NEARUSDT','NMRUSDT','OCEANUSDT','OMGUSDT','OXTUSDT','OGNUSDT','PAXGUSDT','PERPUSDT','PHAUSDT','PLAUSDT','DOTUSDT','MATICUSDT','POWRUSDT','QTUMUSDT','QNTUSDT','RAYUSDT','RENUSDT','RNDRUSDT','REQUSDT','XRPUSDT','SCRTUSDT','KEYUSDT','SRMUSDT','SHIBUSDT','SCUSDT','SOLUSDT','SPELLUSDT','XLMUSDT','GMTUSDT','STORJUSDT','SUSHIUSDT','RADUSDT','FISUSDT','SUPERUSDT','RAREUSDT','SNXUSDT','USTUSDT','TVKUSDT','XTZUSDT','GRTUSDT','SANDUSDT','RUNEUSDT','TUSDT','TRIBEUSDT','TRXUSDT','UNIUSDT','UNFIUSDT','UMAUSDT','USDCUSDT','WAVESUSDT','WOOUSDT','YFIUSDT','ZECUSDT']
tickerPairs = ['BTCUSDT']
loopcount = 0
theta = [0]*3000
theta_0 = 0
success = 0
total = 0
completelist = []
from collections import Counter
from datetime import datetime

API_KEY = 'SAZ4fQcaMP8tf9fMvIm9xUITR5nFGy1ymNIyTjCKdmSrBLApjAd2Np62pezBYEBo'
SECRET_KEY = 'c905nffp3XL4QVKGJ0ag38bxhzSVBfkKefUPXLPujWxH7xZAvPVNeV8E82JmE6ev'
BASE_URL = 'https://api.binance.us'

client = Client(API_KEY, SECRET_KEY, tld='us')
headers = {
    'X-MBX-APIKEY': API_KEY
}


class BinanceException(Exception):
    def __init__(self, status_code, data):

        self.status_code = status_code
        if data:
            self.code = data['code']
            self.msg = data['msg']
        else:
            self.code = None
            self.msg = None
        message = f"{status_code} [{self.code}] {self.msg}"

        # Python 2.x
        # super(BinanceException, self).__init__(message)
        super().__init__(message)
        

stocklist2 = ['META', 'TSLA','BBD','MACK','DIS','NIO','AMZN','AMD','RBLX','GFI','AUY','LCID','AAPL','NU','ITUB','TLRY','AFRM','NVDA','LYFT','PLUG','SNAP','F','GOOGL','DNA','XPEV','CVNA']
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU, RNN
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque
import requests
import os
import numpy as np
import pandas as pd
import random


def buyCrypto(ticker):
    PATH = '/api/v3/order'
    timestamp = int(time.time() * 1000)
    params = {
        'symbol': ticker,
        'side': 'BUY',
        'type': 'LIMIT',
        'timeInForce': 'GTC',
        'quantity': quantity,
        'price': price,
        'recvWindow': 5000,
        'timestamp': timestamp
    }
    
    query_string = urlencode(params)
    params['signature'] = hmac.new(SECRET_KEY.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    
    url = urljoin(BASE_URL, PATH)
    r = requests.post(url, headers=headers, params=params)
    if r.status_code == 200:
        data = r.json()
        print(json.dumps(data, indent=2))
        return r.json()
        
    # else:
    #     raise BinanceException(status_code=r.status_code, data=r.json())

def sellCrypto(ticker):
    PATH = '/api/v3/order'
    timestamp = int(time.time() * 1000)
    params = {
        'symbol': ticker,
        'side': 'SELL',
        'type': 'LIMIT',
        'timeInForce': 'GTC',
        'quantity': quantity,
        'price': price,
        'recvWindow': 5000,
        'timestamp': timestamp
    }
    
    query_string = urlencode(params)
    params['signature'] = hmac.new(SECRET_KEY.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    
    url = urljoin(BASE_URL, PATH)
    r = requests.post(url, headers=headers, params=params)
    if r.status_code == 200:
        data = r.json()
        print(json.dumps(data, indent=2))
    else:
        raise BinanceException(status_code=r.status_code, data=r.json())  
        
def cancelAll(ticker):
    PATH = '/api/v3/openOrders'
    timestamp = int(time.time() * 1000)
    params = {
        'symbol': ticker,
        
        'timestamp': timestamp
    }
    
    query_string = urlencode(params)
    params['signature'] = hmac.new(SECRET_KEY.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    
    url = urljoin(BASE_URL, PATH)
    r = requests.delete(url, headers=headers, params=params)
    if r.status_code == 200:
        data = r.json()
    # else:
    #     raise BinanceException(status_code=r.status_code, data=r.json())  

# # set seed, so we can get the same results after rerunning several times
# np.random.seed(314)
# tf.random.set_seed(314)
# random.seed(314)
# testdump=0

def stonks2(tickabalooza):    
    tf.keras.backend.clear_session()
    def shuffle_in_unison(a, b):
        # shuffle two arrays in the same way
        state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(state)
        np.random.shuffle(b)
   
    def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=True,
                    test_size=0.2, feature_columns=['close', 'volume', 'open', 'high', 'low',"this","that","then","we","yes","maybe" , "open1", "high1",  "low1", "close1",  "volume1","this1","that1","then1","we1","yes1","maybe1"]):
        """
        Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
        Params:
            ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
            n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
            scale (bool): whether to scale prices from 0 to 1, default is True
            shuffle (bool): whether to shuffle the dataset (both training & testing), default is True
            lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
            split_by_date (bool): whether we split the dataset into training/testing by date, setting it
                to False will split datasets in a random way
            test_size (float): ratio for test data, default is 0.2 (20% testing data)
            feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
        """
        # see if ticker is already a loaded stock from yahoo finance
        if isinstance(ticker, str):
            # load it from yahoo_fin library
           # html = 'https://api.tdameritrade.com/v1/marketdata/'+ticker+'/pricehistory?apikey=POH7YLHH0EOOWAHOCJEM0YBXWYIMLWOS&frequencyType=minute&frequency=1&endDate='+str(int(time.time())*1000)+'&startDate='+str(int(time.time())*1000-86400000*7)+'&needExtendedHoursData=false'
           # resp = requests.get(html)
           # atemp1=resp.json()
         #   ttemp1=pd.DataFrame(data=atemp1['candles'])
            if roundOne==0:
                numberOfDays = currentDay+60
                klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, str(numberOfDays) + "days ago UTC")
                klines = pd.DataFrame(data=klines)
                klines=klines.astype(float)
             #   print(klines)
               # klines = klines.drop(columns = [6,7,8,9,10,11])
                klines = klines.rename(columns={0: "datetime" , 1: "open", 2: "high", 3: "low", 4: "close", 5: "volume",6:"this",7:"that",8:"then",9:"we",10:"yes",11:"maybe"})
                
                #print(ttemp1)
                #ttemp1=ttemp1.drop(['datetime'],axis=1)
                klines=klines.set_index('datetime')
                df = klines.copy()
            #    print(df)
                df.drop(df.tail(len(df)-24*60).index,inplace = True)
                
                print('======================================')
                print('stonks2')
                print(df.index[0])
                datetime_obj=datetime.fromtimestamp(float(df.index[0])/1000)
                print(datetime_obj)
                datetime_obj=datetime.fromtimestamp(float(df.index[-1])/1000)
                print(datetime_obj)
                print('--------------------------------------')
                stocklist = ['PAXGUSD', 'ETHUSD','SHIBUSDT','BNBUSDT','FILUSDT','LTCUSDT','SOLUSDT','MATICUSDT','ETHBTC','BTCUSD', 'BTCBUSD']
                # for i in range(0,len(stocklist)):
                #     #print(df)
                #     # time.sleep(2)
                #     if stocklist[i]!=0:
                        
                        
                        
                        
                        
                #         klines = client.get_historical_klines(stocklist[i], Client.KLINE_INTERVAL_1HOUR, str(numberOfDays) + "days ago UTC")
                #         klines = pd.DataFrame(data=klines)
                #         klines=klines.astype(float)
                #         klines = klines.drop(columns = [2,3,4,5,6,7,8,9,10,11])
                #     #    print(klines)
                #         # klines = klines.drop(columns = [6,7,8,9,10,11])
                #         klines = klines.rename(columns={0: "datetime" , 1: "open"+str(i)})
                        
                #         #print(ttemp1)
                #         #ttemp1=ttemp1.drop(['datetime'],axis=1)
                #         klines=klines.set_index('datetime')
                #         df = pd.concat([df,klines],axis=1).sort_index().dropna()
                        
                        
                        
                        
                #         # print(df)
                #         feature_columns.append('open'+str(i))
                        
               # df = pd.concat([df,klines],axis=1).sort_index().dropna()
            #    print(df)
              
                #HERE
               # df.drop(df.tail(1450*7).index,inplace = True)
                
                
                
              #  print(df)
              #  print(df)
              
                # stocklist = ['META', 'TSLA','DIS','NIO','AMZN','AMD','RBLX','AAPL','TLRY','AFRM','NVDA','LYFT','PLUG','SNAP','F','GOOGL','DNA','XPEV','CVNA']
                # stocklist = ['MSFT', 'AAPL', 'TSLA', 'NVDA', 'BAC', 'CSCO', 'AMD', 'NOW', 'INFY', 'F','META', 'CRM', 'SHOP']
                # for i in range(0,len(stocklist)):
                #     #print(df)
                #     # time.sleep(2)
                #     if stocklist[i]!=0:
                #         html = 'https://api.tdameritrade.com/v1/marketdata/'+stocklist[i]+'/pricehistory?apikey=POH7YLHH0EOOWAHOCJEM0YBXWYIMLWOS&frequencyType=minute&frequency=1&endDate='+str(int(time.time())*1000)+'&startDate='+str(int(time.time())*1000-86400000*numberOfDays)+'&needExtendedHoursData=false'
                #         resp = requests.get(html)
                #         atemp1=resp.json()
                #         ttemp1=pd.DataFrame(data=atemp1['candles'])
                #       #   print(ttemp1)
                #         #ttemp1=ttemp1.drop(['datetime'],axis=1)
                #         ttemp1=ttemp1.set_index('datetime')
                #         df2 = ttemp1.copy()
                #         # print(len(df2))
                #         # print(len(df))
    
                #       #   df2=df2.drop('ticker',axis=1)
                #         df2 = df2.rename(columns={'open': 'open'+str(i), 'high': 'high'+str(i),'low':'low'+str(i),'close':'close'+str(i),'volume':'volume'+str(i)})
                #         df3=df2.copy()
                #         # print(df3)
                       
                #         df = pd.concat([df,df3],axis=1).sort_index().dropna()
                #         # print(df)
                #         feature_columns.append('open'+str(i))
                #         feature_columns.append('high'+str(i))
                #         feature_columns.append('low'+str(i))
                #         feature_columns.append('close'+str(i))
                #         feature_columns.append('volume'+str(i))
             #   print(feature_columns)
              #  print(df)
            #    df.to_pickle("storedcases.pkl")
            # else:
            #     df = pd.read_pickle("storedcases.pkl")
            print('hi')
            finalTimeOnData = list(df.index.values)[-1].copy()
            #print((time.time()-list(df.index.values)[-1]/1000)/60)
        elif isinstance(ticker, pd.DataFrame):
            # already loaded, use it directly
            df = ticker
        else:
            raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
        # this will contain all the elements we want to return from this function
        result = {}
        # we will also return the original dataframe itself
        result['df'] = df.copy()
        # make sure that the passed feature_columns exist in the dataframe
        for col in feature_columns:
            assert col in df.columns, f"'{col}' does not exist in the dataframe."
        # add date as a column
        if "date" not in df.columns:
            df["date"] = df.index
        if scale:
            column_scaler = {}
            # scale the data (prices) from 0 to 1
            for column in feature_columns:
                scaler = preprocessing.MinMaxScaler()
                df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
                column_scaler[column] = scaler
            # add the MinMaxScaler instances to the result returned
            result["column_scaler"] = column_scaler
        # add the target column (label) by shifting by `lookup_step`
        df['future'] = df['close'].shift(-lookup_step)
        # last `lookup_step` columns contains NaN in future column
        # get them before droping NaNs
        last_sequence = np.array(df[feature_columns].tail(lookup_step))
        # drop NaNs
        df.dropna(inplace=True)
        sequence_data = []
        sequences = deque(maxlen=n_steps)
        for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
            sequences.append(entry)
            if len(sequences) == n_steps:
                sequence_data.append([np.array(sequences), target])
      
        # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
        # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
        # this last_sequence will be used to predict future stock prices that are not available in the dataset
        last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
        last_sequence = np.array(last_sequence).astype(np.float32)
        # add to result
        result['last_sequence'] = last_sequence
        # construct the X's and y's
        X, y = [], []
        for seq, target in sequence_data:
            X.append(seq)
            y.append(target)
        # convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        if split_by_date:
            # split the dataset into training & testing sets by date (not randomly splitting)
            train_samples = int((1 - test_size) * len(X))
            result["X_train"] = X[:train_samples]
            result["y_train"] = y[:train_samples]
            result["X_test"]  = X[train_samples:]
            result["y_test"]  = y[train_samples:]
            if shuffle:
               
                # shuffle the datasets for training (if shuffle parameter is set)
                shuffle_in_unison(result["X_train"], result["y_train"])
                shuffle_in_unison(result["X_test"], result["y_test"])
        else:    
            # split the dataset randomly
            result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                    test_size=test_size, shuffle=shuffle)
        # get the list of test set dates
        dates = result["X_test"][:, -1, -1]
        # retrieve test features from the original dataframe
        result["test_df"] = result["df"].loc[dates]
        # remove duplicated dates in the testing dataframe
        result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
        # remove dates from the training/testing sets & convert to float32
        result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
        result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)
       # print(result)
        return result, finalTimeOnData, feature_columns
       
    import os
    import time
    from tensorflow.keras.layers import LSTM,GRU, RNN
   
    def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                    loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
        model = Sequential()
        for i in range(n_layers):
            if i == 0:
                # first layer
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
                else:
                    model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
            elif i == n_layers - 1:
                # last layer
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=False)))
                else:
                    model.add(cell(units, return_sequences=False))
            else:
                # hidden layers
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=True)))
                else:
                    model.add(cell(units, return_sequences=True))
            # add dropout after each layer
            model.add(Dropout(dropout))
        model.add(Dense(1, activation="tanh"))
        model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
        return model

   

    # Window size or the sequence length
    N_STEPS = 4
    # Lookup step, 1 is the next day
    LOOKUP_STEP = lookItUp
    # whether to scale feature columns & output price as well
    SCALE = True
    scale_str = f"sc-{int(SCALE)}"
    # whether to shuffle the dataset
    SHUFFLE = False
    shuffle_str = f"sh-{int(SHUFFLE)}"
    # whether to split the training/testing set by date
    SPLIT_BY_DATE = False
    split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"
    # test ratio size, 0.2 is 20%
    TEST_SIZE = 0.1
    # features to us
    FEATURE_COLUMNS = ["close", "volume", "open", "high", "low","this","that","then","we","yes","maybe"]
    # date now
    date_now = time.strftime("%Y-%m-%d")
    ### model parameters
    N_LAYERS = 4
    # LSTM cell
    CELL = GRU
    # 256 LSTM neurons
    UNITS = 8
    # 40% dropout
    DROPOUT = 0.2
    # whether to use bidirectional RNNs
    BIDIRECTIONAL = False
    ### training parameters
    # mean absolute error loss
    # LOSS = "mae"
    # huber loss
    LOSS = "huber_loss"
    OPTIMIZER = "adam"
    BATCH_SIZE = 32
    EPOCHS = 1
    # Amazon stock market
    ticker = tickabalooza
    ticker_data_filename = os.path.join("data", f"{ticker}_.csv")
    # model name to save, making it as unique as possible based on parameters
    model_name = f"_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-\
    {LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
    if BIDIRECTIONAL:
        model_name += "-b"
       
       
    # create these folders if they does not exist
    if not os.path.isdir("results"):
        os.mkdir("results")
    if not os.path.isdir("logs"):
        os.mkdir("logs")
    if not os.path.isdir("data"):
        os.mkdir("data")
    # if 365-currentDay>2:   
    #     os.remove(os.path.join("results", model_name + ".h5"))   
    # load the data
    data, finalTimeInfo,FEATURE_COLUMNS = load_data(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE,
                    shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                    feature_columns=FEATURE_COLUMNS)
    #print(data)
    #print(data)
    #print(list(data.index.values)[-1])
    # save the dataframe
    data["df"].to_csv(ticker_data_filename)
    # construct the model
    model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                        dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
    # some tensorflow callbacks
    checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=0)
    tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
    # train the model and save the weights whenever we see
    # a new optimal model using ModelCheckpoint
    #os.remove(os.path.join("results", model_name + ".h5"))
    early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    mode='min', 
    restore_best_weights=True)
    if roundOne==1:
        model_path = os.path.join("results", model_name) + ".h5"
        model.load_weights(model_path)
    
    history = model.fit(data["X_train"], data["y_train"],
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                 #       validation_data=(data["X_test"], data["y_test"]),
                        callbacks=[checkpointer, tensorboard],
                        validation_split=0.25,
                        verbose=1)
   
    import matplotlib.pyplot as plt
   
    def plot_graph(test_df):
        """
        This function plots true close price along with predicted close price
        with blue and red colors respectively
        """
        plt.plot(test_df[f'true_close_{LOOKUP_STEP}'], c='b')
        plt.plot(test_df[f'close_{LOOKUP_STEP}'], c='r')
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend(["Actual Price", "Predicted Price"])
        plt.show()
       
    def get_final_df(model, data):
        """
        This function takes the `model` and `data` dict to
        construct a final dataframe that includes the features along
        with true and predicted prices of the testing dataset
        """
        # if predicted future price is higher than the current,
        # then calculate the true future price minus the current price, to get the buy profit
        
        buy_profit  = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
        # if the predicted future price is lower than the current price,
        # then subtract the true future price from the current price
        sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0
        X_test = data["X_test"]
        y_test = data["y_test"]
        # perform prediction and get prices
        y_pred = model.predict(X_test)
        if SCALE:
            y_test = np.squeeze(data["column_scaler"]["close"].inverse_transform(np.expand_dims(y_test, axis=0)))
            y_pred = np.squeeze(data["column_scaler"]["close"].inverse_transform(y_pred))
        test_df = data["test_df"]
        # add predicted future prices to the dataframe
        test_df[f"close_{LOOKUP_STEP}"] = y_pred
        # add true future prices to the dataframe
        test_df[f"true_close_{LOOKUP_STEP}"] = y_test
        # sort the dataframe by date
        test_df.sort_index(inplace=True)
        final_df = test_df
        # add the buy profit column
        final_df["buy_profit"] = list(map(buy_profit,
                                        final_df["close"],
                                        final_df[f"close_{LOOKUP_STEP}"],
                                        final_df[f"true_close_{LOOKUP_STEP}"])
                                        # since we don't have profit for last sequence, add 0's
                                        )
        # add the sell profit column
        final_df["sell_profit"] = list(map(sell_profit,
                                        final_df["close"],
                                        final_df[f"close_{LOOKUP_STEP}"],
                                        final_df[f"true_close_{LOOKUP_STEP}"])
                                        # since we don't have profit for last sequence, add 0's
                                        )
        return final_df
   
   
    def predict(model, data):
        # retrieve the last sequence from data
        last_sequence = data["last_sequence"][-N_STEPS:]
        # expand dimension
        last_sequence = np.expand_dims(last_sequence, axis=0)
        # get the prediction (scaled from 0 to 1)
        prediction = model.predict(last_sequence)
        # get the price (by inverting the scaling)
        if SCALE:
            predicted_price = data["column_scaler"]["close"].inverse_transform(prediction)[0][0]
        else:
            predicted_price = prediction[0][0]
        return predicted_price
   
   
    # load optimal model weights from results folder
    model_path = os.path.join("results", model_name) + ".h5"
    model.load_weights(model_path)
   
   
    # evaluate the model
    loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    # calculate the mean absolute error (inverse scaling)
    if SCALE:
        mean_absolute_error = data["column_scaler"]["close"].inverse_transform([[mae]])[0][0]
    else:
        mean_absolute_error = mae
       
    # get the final dataframe for the testing set
    final_df = get_final_df(model, data)
   
   
    # predict the future price
    future_price = predict(model, data)
   
   
   
    # we calculate the accuracy by counting the number of positive profits
    accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buy_profit'] > 0])) / len(final_df)
    # calculating total buy & sell profit
    total_buy_profit  = final_df["buy_profit"].sum()
    total_sell_profit = final_df["sell_profit"].sum()
    # total profit by adding sell & buy together
    total_profit = total_buy_profit + total_sell_profit
    # dividing total profit by number of testing samples (number of trades)
    profit_per_trade = total_profit / len(final_df)
   
   
    # printing metrics
    # print(f"Future price after {LOOKUP_STEP} days is {future_price}$")
    # print(f"{LOSS} loss:", loss)
    # print("Mean Absolute Error:", mean_absolute_error)
    # print("Accuracy score:", accuracy_score)
    # print("Total buy profit:", total_buy_profit)
    # print("Total sell profit:", total_sell_profit)
    # print("Total profit:", total_profit)
    # print("Profit per trade:", profit_per_trade)




def stonks(tickabalooza):    
    tf.keras.backend.clear_session()
    def shuffle_in_unison(a, b):
        # shuffle two arrays in the same way
        state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(state)
        np.random.shuffle(b)
   
    def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=True,
                    test_size=0.2, feature_columns=['close', 'volume', 'open', 'high', 'low',"this","that","then","we","yes","maybe"]):
        """
        Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
        Params:
            ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
            n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
            scale (bool): whether to scale prices from 0 to 1, default is True
            shuffle (bool): whether to shuffle the dataset (both training & testing), default is True
            lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
            split_by_date (bool): whether we split the dataset into training/testing by date, setting it
                to False will split datasets in a random way
            test_size (float): ratio for test data, default is 0.2 (20% testing data)
            feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
        """
        # see if ticker is already a loaded stock from yahoo finance
        if isinstance(ticker, str):
            # load it from yahoo_fin library
           # html = 'https://api.tdameritrade.com/v1/marketdata/'+ticker+'/pricehistory?apikey=POH7YLHH0EOOWAHOCJEM0YBXWYIMLWOS&frequencyType=minute&frequency=1&endDate='+str(int(time.time())*1000)+'&startDate='+str(int(time.time())*1000-86400000*7)+'&needExtendedHoursData=false'
           # resp = requests.get(html)
           # atemp1=resp.json()
         #   ttemp1=pd.DataFrame(data=atemp1['candles'])
            numberOfDays = currentDay+1
            klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, str(numberOfDays) + "days ago UTC")
            klines = pd.DataFrame(data=klines)
            klines=klines.astype(float)
         #   print(klines)
           # klines = klines.drop(columns = [6,7,8,9,10,11])
            klines = klines.rename(columns={0: "datetime" , 1: "open", 2: "high", 3: "low", 4: "close", 5: "volume",6:"this",7:"that",8:"then",9:"we",10:"yes",11:"maybe"})
            
            #print(ttemp1)
            #ttemp1=ttemp1.drop(['datetime'],axis=1)
            klines=klines.set_index('datetime')
            df = klines.copy()
        #   print(df)
            df.drop(df.tail(len(df)-2*24).index,inplace = True)
            print('stonks1')
            datetime_obj=datetime.fromtimestamp(float(df.index[0])/1000)
            print(datetime_obj)
            datetime_obj=datetime.fromtimestamp(float(df.index[-1])/1000)
            print(datetime_obj)
        #     klines = client.get_historical_klines("ETHUSD", Client.KLINE_INTERVAL_1MINUTE, str(numberOfDays) + "days ago UTC")
        #     klines = pd.DataFrame(data=klines)
        #     klines=klines.astype(float)
        #     print(klines)
        #    # klines = klines.drop(columns = [6,7,8,9,10,11])
        #     klines = klines.rename(columns={0: "datetime" , 1: "open1", 2: "high1", 3: "low1", 4: "close1", 5: "volume1",6:"this1",7:"that1",8:"then1",9:"we1",10:"yes1",11:"maybe1"})
            
        #     #print(ttemp1)
        #     #ttemp1=ttemp1.drop(['datetime'],axis=1)
        #     klines=klines.set_index('datetime')
        #     df = pd.concat([df,klines],axis=1).sort_index().dropna()
        # #    print(df)
          #  df.drop(df.tail(1450).index,inplace = True)
            
            
            
          #  print(df)
          #  print(df)
          
            stocklist = ['META', 'TSLA','DIS','NIO','AMZN','AMD','RBLX','AAPL','TLRY','AFRM','NVDA','LYFT','PLUG','SNAP','F','GOOGL','DNA','XPEV','CVNA']
            stocklist = ['MSFT', 'AAPL', 'TSLA', 'NVDA', 'BAC', 'CSCO', 'AMD', 'NOW', 'INFY', 'F','META', 'CRM', 'SHOP']
            
            
            
            
            
            stocklist = ['PAXGUSD', 'ETHUSD','SHIBUSDT','BNBUSDT','FILUSDT','LTCUSDT','SOLUSDT','MATICUSDT','ETHBTC','BTCUSD','BTCBUSD' ]
            # for i in range(0,len(stocklist)):
            #     #print(df)
            #     # time.sleep(2)
            #     if stocklist[i]!=0:
                    
                    
                    
                    
                    
            #         klines = client.get_historical_klines(stocklist[i], Client.KLINE_INTERVAL_1HOUR, str(numberOfDays) + "days ago UTC")
            #         klines = pd.DataFrame(data=klines)
            #         klines=klines.astype(float)
            #         klines = klines.drop(columns = [2,3,4,5,6,7,8,9,10,11])
            #      #   print(klines)
            #         # klines = klines.drop(columns = [6,7,8,9,10,11])
            #         klines = klines.rename(columns={0: "datetime" , 1: "open"+str(i)})
                    
            #         #print(ttemp1)
            #         #ttemp1=ttemp1.drop(['datetime'],axis=1)
            #         klines=klines.set_index('datetime')
            #         df = pd.concat([df,klines],axis=1).sort_index().dropna()
                    
                    
                    
                    
            #         # print(df)
            #         feature_columns.append('open'+str(i))
                    
             #  print(df)
            dfc1=df.copy()
            dfc1.drop(dfc1.tail(24).index,inplace = True)
            
            
            print('weight day')
            datetime_obj=datetime.fromtimestamp(float(dfc1.index[0])/1000)
            print(datetime_obj)
            datetime_obj=datetime.fromtimestamp(float(dfc1.index[-1])/1000)
            print(datetime_obj)
            
            df.drop(df.head(24).index,inplace = True)   
            
            print('test day')
            datetime_obj=datetime.fromtimestamp(float(df.index[0])/1000)
            print(datetime_obj)
            datetime_obj=datetime.fromtimestamp(float(df.index[-1])/1000)
            print(datetime_obj)
            
          #  print(df)
            finalTimeOnData = list(df.index.values)[-1].copy()
            #print((time.time()-list(df.index.values)[-1]/1000)/60)
        elif isinstance(ticker, pd.DataFrame):
            # already loaded, use it directly
            df = ticker
        else:
            raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
        # this will contain all the elements we want to return from this function
        result = {}
        # we will also return the original dataframe itself
        result['df'] = df.copy()
        # make sure that the passed feature_columns exist in the dataframe
        for col in feature_columns:
            assert col in df.columns, f"'{col}' does not exist in the dataframe."
        # add date as a column
        if "date" not in df.columns:
            df["date"] = df.index
        if scale:
            column_scaler = {}
            # scale the data (prices) from 0 to 1
            for column in feature_columns:
                scaler = preprocessing.MinMaxScaler()
                scaler.fit(np.expand_dims(dfc1[column].values, axis=1))
                df[column] = scaler.transform(np.expand_dims(df[column].values, axis=1))
                column_scaler[column] = scaler
            # add the MinMaxScaler instances to the result returned
            result["column_scaler"] = column_scaler
        # add the target column (label) by shifting by `lookup_step`
        df['future'] = df['close'].shift(-lookup_step)
        # last `lookup_step` columns contains NaN in future column
        # get them before droping NaNs
        last_sequence = np.array(df[feature_columns].tail(lookup_step))
        # drop NaNs
        df.dropna(inplace=True)
        sequence_data = []
        sequences = deque(maxlen=n_steps)
        for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
            sequences.append(entry)
            if len(sequences) == n_steps:
                sequence_data.append([np.array(sequences), target])
        # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
        # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
        # this last_sequence will be used to predict future stock prices that are not available in the dataset
        last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
        last_sequence = np.array(last_sequence).astype(np.float32)
        # add to result
        result['last_sequence'] = last_sequence
        # construct the X's and y's
        X, y = [], []
        for seq, target in sequence_data:
            X.append(seq)
            y.append(target)
        # convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        if split_by_date:
            # split the dataset into training & testing sets by date (not randomly splitting)
            train_samples = int((1 - test_size) * len(X))
            result["X_train"] = X[:train_samples]
            result["y_train"] = y[:train_samples]
            result["X_test"]  = X[train_samples:]
            result["y_test"]  = y[train_samples:]
            if shuffle:
               
                # shuffle the datasets for training (if shuffle parameter is set)
                shuffle_in_unison(result["X_train"], result["y_train"])
                shuffle_in_unison(result["X_test"], result["y_test"])
        else:    
            # split the dataset randomly
            result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                    test_size=test_size, shuffle=shuffle)
        # get the list of test set dates
        dates = result["X_test"][:, -1, -1]
        # retrieve test features from the original dataframe
        result["test_df"] = result["df"].loc[dates]
        # remove duplicated dates in the testing dataframe
        result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
        # remove dates from the training/testing sets & convert to float32
        result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
        result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)
       # print(result)
        return result, finalTimeOnData,feature_columns
       
    import os
    import time
    from tensorflow.keras.layers import LSTM,GRU, RNN
   
    def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                    loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
        model = Sequential()
        for i in range(n_layers):
            if i == 0:
                # first layer
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
                else:
                    model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
            elif i == n_layers - 1:
                # last layer
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=False)))
                else:
                    model.add(cell(units, return_sequences=False))
            else:
                # hidden layers
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=True)))
                else:
                    model.add(cell(units, return_sequences=True))
            # add dropout after each layer
            model.add(Dropout(dropout))
        model.add(Dense(1, activation="tanh"))
        model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
        return model

   
  
    # Window size or the sequence length
    N_STEPS = 4
    # Lookup step, 1 is the next day
    LOOKUP_STEP = lookItUp
    # whether to scale feature columns & output price as well
    SCALE = True
    scale_str = f"sc-{int(SCALE)}"
    # whether to shuffle the dataset
    SHUFFLE = False
    shuffle_str = f"sh-{int(SHUFFLE)}"
    # whether to split the training/testing set by date
    SPLIT_BY_DATE = False
    split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"
    # test ratio size, 0.2 is 20%
    TEST_SIZE = 0.8
    # features to us
    FEATURE_COLUMNS = ["close", "volume", "open", "high", "low","this","that","then","we","yes","maybe"]
    # date now
    date_now = time.strftime("%Y-%m-%d")
    ### model parameters
    N_LAYERS = 4
    # LSTM cell
    CELL = GRU
    # 256 LSTM neurons
    UNITS = 8
    # 40% dropout
    DROPOUT = 0.2
    # whether to use bidirectional RNNs
    BIDIRECTIONAL = False
    ### training parameters
    # mean absolute error loss
    # LOSS = "mae"
    # huber loss
    LOSS = "huber_loss"
    OPTIMIZER = "adam"
    BATCH_SIZE = 32
    EPOCHS = 1
    # Amazon stock market
    ticker = tickabalooza
    ticker_data_filename = os.path.join("data", f"{ticker}_.csv")
    # model name to save, making it as unique as possible based on parameters
    model_name = f"_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-\
    {LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
    if BIDIRECTIONAL:
        model_name += "-b"
       
       
    # create these folders if they does not exist
    if not os.path.isdir("results"):
        os.mkdir("results")
    if not os.path.isdir("logs"):
        os.mkdir("logs")
    if not os.path.isdir("data"):
        os.mkdir("data")
       
       
    # load the data
    data, finalTimeInfo,FEATURE_COLUMNS = load_data(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE,
                    shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                    feature_columns=FEATURE_COLUMNS)
    #print(data)
    #print(data)
    #print(list(data.index.values)[-1])
    # save the dataframe
    data["df"].to_csv(ticker_data_filename)
    # construct the model
    print(len(FEATURE_COLUMNS))
    model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                        dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
    # some tensorflow callbacks
    checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=0)
    tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
    # train the model and save the weights whenever we see
    # a new optimal model using ModelCheckpoint
    
    early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    mode='min', 
    restore_best_weights=True)

    model_path = os.path.join("results", model_name) + ".h5"
    model.load_weights(model_path)
    
    # history = model.fit(data["X_train"], data["y_train"],
    #                     batch_size=BATCH_SIZE,
    #                     epochs=EPOCHS,
    #              #       validation_data=(data["X_test"], data["y_test"]),
    #                     callbacks=[checkpointer, tensorboard],
    #                     validation_split=0.25,
    #                     verbose=1)
   
    import matplotlib.pyplot as plt
   
    def plot_graph(test_df):
        """
        This function plots true close price along with predicted close price
        with blue and red colors respectively
        """
        plt.plot(test_df[f'true_close_{LOOKUP_STEP}'], c='b')
        plt.plot(test_df[f'close_{LOOKUP_STEP}'], c='r')
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend(["Actual Price", "Predicted Price"])
        plt.show()
       
    def get_final_df(model, data):
        """
        This function takes the `model` and `data` dict to
        construct a final dataframe that includes the features along
        with true and predicted prices of the testing dataset
        """
        # if predicted future price is higher than the current,
        # then calculate the true future price minus the current price, to get the buy profit
        
        buy_profit  = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
        # if the predicted future price is lower than the current price,
        # then subtract the true future price from the current price
        sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0
        X_test = data["X_test"]
        y_test = data["y_test"]
        # perform prediction and get prices
        y_pred = model.predict(X_test)
        if SCALE:
            y_test = np.squeeze(data["column_scaler"]["close"].inverse_transform(np.expand_dims(y_test, axis=0)))
            y_pred = np.squeeze(data["column_scaler"]["close"].inverse_transform(y_pred))
        test_df = data["test_df"]
        # add predicted future prices to the dataframe
        test_df[f"close_{LOOKUP_STEP}"] = y_pred
        # add true future prices to the dataframe
        test_df[f"true_close_{LOOKUP_STEP}"] = y_test
        # sort the dataframe by date
        test_df.sort_index(inplace=True)
        final_df = test_df
        # add the buy profit column
        final_df["buy_profit"] = list(map(buy_profit,
                                        final_df["close"],
                                        final_df[f"close_{LOOKUP_STEP}"],
                                        final_df[f"true_close_{LOOKUP_STEP}"])
                                        # since we don't have profit for last sequence, add 0's
                                        )
        # add the sell profit column
        final_df["sell_profit"] = list(map(sell_profit,
                                        final_df["close"],
                                        final_df[f"close_{LOOKUP_STEP}"],
                                        final_df[f"true_close_{LOOKUP_STEP}"])
                                        # since we don't have profit for last sequence, add 0's
                                        )
        return final_df
   
   
    def predict(model, data):
        # retrieve the last sequence from data
        last_sequence = data["last_sequence"][-N_STEPS:]
        # expand dimension
        last_sequence = np.expand_dims(last_sequence, axis=0)
        # get the prediction (scaled from 0 to 1)
        prediction = model.predict(last_sequence)
        # get the price (by inverting the scaling)
        if SCALE:
            predicted_price = data["column_scaler"]["close"].inverse_transform(prediction)[0][0]
        else:
            predicted_price = prediction[0][0]
        return predicted_price
   
   
    # load optimal model weights from results folder
    model_path = os.path.join("results", model_name) + ".h5"
    model.load_weights(model_path)
   
   
    # evaluate the model
    loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    # calculate the mean absolute error (inverse scaling)
    if SCALE:
        mean_absolute_error = data["column_scaler"]["close"].inverse_transform([[mae]])[0][0]
    else:
        mean_absolute_error = mae
       
    # get the final dataframe for the testing set
    final_df = get_final_df(model, data)
   
   
    # predict the future price
    future_price = predict(model, data)
   
   
   
    # we calculate the accuracy by counting the number of positive profits
    accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buy_profit'] > 0])) / len(final_df)
    # calculating total buy & sell profit
    total_buy_profit  = final_df["buy_profit"].sum()
    total_sell_profit = final_df["sell_profit"].sum()
    # total profit by adding sell & buy together
    total_profit = total_buy_profit + total_sell_profit
    # dividing total profit by number of testing samples (number of trades)
    profit_per_trade = total_profit / len(final_df)
   
   
    # printing metrics
    print(f"Future price after {LOOKUP_STEP} days is {future_price}$")
    print(f"{LOSS} loss:", loss)
    print("Mean Absolute Error:", mean_absolute_error)
    print("Accuracy score:", accuracy_score)
    print("Total buy profit:", total_buy_profit)
    print("Total sell profit:", total_sell_profit)
    print("Total profit:", total_profit)
    print("Profit per trade:", profit_per_trade)
    
    
    return accuracy_score,future_price, total_buy_profit, finalTimeInfo

def getLast1000Trades(ii):
    resp = requests.get('https://www.binance.com/api/v3/trades?symbol='+ii)
    return resp.json()

currentDay=31
start=currentDay
finalshowdown = []
import time
tanky=0
avgacc=[]
##number of days equals time delay
lookItUp = 1
counting40 = []
t40=[]
countingEverything = 0
buyprofac=[]
roundOne=0
countacc=0
countloops =0 
if roundOne==0:
    stonks2('AAPl')
    roundOne=0
import datetime
now = datetime.datetime.now()
from datetime import datetime
avgacc=[]
dollars=1000
dollararr=[]
negativedays=0
negativedaycounter=0
while currentDay>1:

    # if now.hour==1:
    #     time.sleep(3*3600)
    
    
        #timeCheck1 = time.time()
     #   lookItUp = 30
     #   nsteps = 120
        stonks2('AAPl')
        acc,guess,buyProf, finalTimeData = stonks('AAPL')
        avgacc.append(buyProf/guess-0.1/100)
        dollararr.append(dollars)
        if len(dollararr)>2:
            if dollararr[-1]-dollararr[-2]<0:
                negativedaycounter-=1
            else:
                negativedaycounter=0
        if negativedaycounter<negativedays:
            negativedays=negativedaycounter
            
        print('max days in red:',negativedays)
        
        dollars=dollars+dollars*avgacc[-1]
        currentDay-=1
        print('$',dollars)
        print(start-currentDay)
        print(np.mean(np.array(avgacc)),'-------------')
import matplotlib.pyplot as plt

plt.plot(dollararr)
plt.ylabel('some numbers')
plt.show()
        
