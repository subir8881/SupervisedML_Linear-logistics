#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd 
import numpy as np
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import matplotlib.pylab as plt


# In[9]:


startup = pd.read_csv('50_Startups.csv')
startup.head()


# In[10]:


startup.corr()


# In[13]:


sns.pairplot(startup)


# In[15]:


startup1 = startup.rename({'R&D Spend': 'RD'}, axis= 1)


# In[16]:


startup2 = startup1.rename({'Marketing Spend': 'Marketing'}, axis= 1)


# In[17]:


import statsmodels.formula.api as smf 
startup3 = smf.ols('Profit~Administration+Marketing+State+RD',data=startup2).fit() 


# In[18]:


startup3.summary()


# In[19]:


startupR=smf.ols('Profit~Marketing',data = startup2).fit()  
startupR.summary()


# In[20]:


startupR=smf.ols('Profit~RD',data = startup2).fit()  
startupR.summary()


# In[21]:


startupR=smf.ols('Profit~Administration',data = startup2).fit()  
startupR.summary()


# In[22]:


startupR=smf.ols('Profit~Marketing+Administration',data = startup2).fit()  
startupR.summary()


# In[23]:


startupR=smf.ols('Profit~State',data = startup2).fit()  
startupR.summary()


# In[29]:


rsq_RD = smf.ols('RD~Administration+Marketing+State',data=startup2).fit().rsquared  
vif_RD = 1/(1-rsq_RD)


# In[28]:


rsq_Administration = smf.ols('Administration~RD+Marketing+State',data=startup2).fit().rsquared  
vif_Administration = 1/(1-rsq_Administration)


# In[27]:


rsq_Marketing = smf.ols('Marketing~RD+Administration+State',data=startup2).fit().rsquared  
vif_Marketing = 1/(1-rsq_Marketing)


# In[26]:


rsq_State = smf.ols('State~RD+Administration+Marketing',data=startup4).fit().rsquared  
vif_State = 1/(1-rsq_State)


# In[33]:


STARTUP = {'Variables':['Administration','Marketing','RD'],'VIF':[vif_RD,vif_Administration,vif_Marketing]}
Vif_frame = pd.DataFrame(STARTUP)  
Vif_frame


# In[34]:


res=startup3.resid
res


# In[35]:


import statsmodels.api as sm
qqplot=sm.qqplot(res, line='q')
plt.title("TEST FOR NORAMLITY OF RESIDUALS (Q-Q plot)")
plt.show


# In[36]:


list(np.where(startup3.resid>10))


# In[37]:


def get_standardized_values(vals) :
    return (vals - vals.mean())/vals.std()


# In[39]:


plt.scatter(get_standardized_values(startup3.fittedvalues),
           get_standardized_values(startup3.resid))
plt.title("Resdial Plot")
plt.xlabel('standardized fitted values')
plt.ylabel('standardized residual values')
plt.show()


# In[40]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(startup3, "RD", fig=fig)


# In[41]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(startup3, "Marketing", fig=fig)


# In[42]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(startup3, "Administration", fig=fig)


# In[44]:


model_influence = startup3.get_influence()


# In[45]:


(c,_) = model_influence.cooks_distance


# In[46]:


c


# In[54]:


fig = plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(startup)), np.round(c,3))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')


# In[55]:


np.argmax(c), np.max(c)


# In[56]:


sm.graphics.influence_plot(startup3)


# In[58]:


k = startup2.shape[1]
n = startup2.shape[0]
leverage_cuttoff = 3*((k+1)/n)


# In[59]:


leverage_cuttoff


# In[69]:


startup[startup.index.isin([48,46,49])]


# In[67]:


startup.head()


# In[72]:


#Improving the Model
startup_new= startup.drop(startup.index[[48,49,46]],axis=0).reset_index()


# In[73]:


startup_new


# In[74]:


startup_N=startup_new.drop(['index'], axis=1)


# In[75]:


startup_N


# In[83]:


sm.graphics.plot_partregress_grid(startup3)
                                


# In[ ]:


final_STARTUP= smf.ols('Profit~RD+Marketing+State+Administration',data = startup3).fit()

