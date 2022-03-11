#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install yfinance


# In[3]:


import yfinance as yf
import numpy as np
import pandas as pd


# In[4]:


data = yf.download("ADA-USD", start="2021-03-01", end="2022-03-01")


# In[27]:


data.head()


# In[28]:


data.tail()


# In[18]:


data['Adj Close'].head()


# In[19]:


data['Adj Close'].plot(figsize=(10, 12), subplots=True)


# In[20]:


import numpy as np


# In[22]:


normal_return = []
for i in range(0,len(data)-1):
    adjclose_yesterday = data.iloc[i]['Adj Close']
    adjclose_today = data.iloc[i+1]['Adj Close']
    x = (adjclose_today - adjclose_yesterday) / adjclose_yesterday
    normal_return.append(x)
normal_return[:5]


# In[24]:


log_return = []
for i in range(0,len(data)-1):
    adjclose_yesterday = data.iloc[i]['Adj Close']
    adjclose_today = data.iloc[i+1]['Adj Close']
    y = np.log(adjclose_today / adjclose_yesterday)
    log_return.append(y)
log_return[:5]


# In[25]:


dfnr = pd.DataFrame(normal_return, columns = ['normal']) 
nr = dfnr.mean() * len(dfnr)
nv = dfnr.std() * (len(dfnr) ** 0.5)
print('The annulized normal return is %.8f and its annulized volatility is %.8f' % (nr,nv))


# In[29]:


import os


# In[30]:


S0 =0.959886         # spot stock price
K = 2.00              # strike
T = 1.0                 # maturity 
r = 0.0114                # risk free rate 
sig = 1.132               # diffusion coefficient or volatility
N = 3                   # number of periods or number of time steps  
payoff = "put"          # payoff 


# In[31]:


dT = float(T) / N                             # Delta t
u = np.exp(sig * np.sqrt(dT))                 # up factor
d = 1.0 / u             


# In[32]:


S = np.zeros((N + 1, N + 1))
S[0, 0] = S0
z = 1
for t in range(1, N + 1):
    for i in range(z):
        S[i, t] = S[i, t-1] * u
        S[i+1, t] = S[i, t-1] * d
    z += 1


# In[33]:


S


# In[34]:


a = np.exp(r * dT)    # risk free compound return
p = (a - d)/ (u - d)  # risk neutral up probability
q = 1.0 - p           # risk neutral down probability
p


# In[36]:


S_T = S[:,-1]
V = np.zeros((N + 1, N + 1))
if payoff =="call":
    V[:,-1] = np.maximum(S_T-K, 0.0)
elif payoff =="put":
    V[:,-1] = np.maximum(K-S_T, 0.0)
V


# In[37]:


# for European Option
for j in range(N-1, -1, -1):
    for i in range(j+1):
        V[i,j] = np.exp(-r*dT) * (p * V[i,j + 1] + q * V[i + 1,j + 1])
V


# In[40]:


print('European ' + payoff, str( V[0,0]))


# In[41]:


# for American Option
if payoff =="call":
    for j in range(N-1, -1, -1):
        for i in range(j+1):
            V[i,j] = np.maximum(S[i,j] - K,np.exp(-r*dT) * (p * V[i,j + 1] + q * V[i + 1,j + 1]))
elif payoff =="put":
    for j in range(N-1, -1, -1):
        for i in range(j+1):
            V[i,j] = np.maximum(K - S[i,j],np.exp(-r*dT) * (p * V[i,j + 1] + q * V[i + 1,j + 1]))
V


# In[42]:


print('American ' + payoff, str( V[0,0]))


# In[43]:


def mcs_simulation_np(p):
    M = p
    I = p
    dt = T / M 
    S = np.zeros((M + 1, I))
    S[0] = S0 
    rn = np.random.standard_normal(S.shape) 
    for t in range(1, M + 1): 
        S[t] = S[t-1] * np.exp((r - sigma ** 2 / 2) * dt + sigma * np.sqrt(dt) * rn[t]) 
    return S


# In[44]:


T = 1
r = 0.0114
sigma = 1.132
S0 = 0.959886
K = 2.00  


# In[45]:


S = mcs_simulation_np(1000)


# In[46]:


S = np.transpose(S)
S


# In[49]:


import matplotlib.pyplot as plt
n, bins, patches = plt.hist(x=S[:,-1], bins='auto', color='#32cd32',alpha=0.7, rwidth=0.85)

plt.grid(axis='y', alpha=0.75)
plt.xlabel('S_T')
plt.ylabel('Frequency')
plt.title('Frequency distribution of the simulated end-of-preiod values')


# In[50]:


p = np.mean(np.maximum(K - S[:,-1],0))
print('European put', str(p))


# In[51]:


c = np.mean(np.maximum(S[:,-1] - K,0))
print('European call', str(c))


# In[ ]:




