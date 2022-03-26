#!/usr/bin/env python
# coding: utf-8

# In[150]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[151]:


startup1= pd.read_csv('50_Startups.csv')


# In[152]:



sns.pairplot(startup1)


# In[153]:


startup1.head()


# In[154]:


startup1.corr()


# In[155]:


sns.barplot(x='State',y='Profit', data=startup1, palette="Blues_d")


# In[156]:


startup2 = startup1.rename({'R&D Spend': 'RD'}, axis= 1)
startup3 = startup1.rename({'Marketing Spend': 'Marketing'}, axis= 1)


# In[157]:


startup3.head()


# In[ ]:





# In[158]:


from sklearn.model_selection import train_test_split


# In[159]:


startups_dum = pd.get_dummies(startup3,drop_first =True)  


# In[160]:


startup_train,startup_test = train_test_split(startups_dum,test_size = 0.3)


# In[161]:


startup_train.corr


# In[162]:


startup_train.columns= ["RD","administration","Marketing","profit","state_florida","state_newyork"]
startup_test.columns= ["RD","administration","Marketing","profit","state_florida","state_newyork"]


# In[163]:


import seaborn as sb


# In[164]:


sb.pairplot(startup_train[["RD","administration","profit", "Marketing", "state_florida","state_newyork"]])


# In[165]:


import statsmodels.formula.api as smf


# In[166]:


model1= smf.ols("profit ~ RD+administration+Marketing+state_florida+state_newyork",data = startup_train).fit()


# In[167]:


model1.summary()


# In[168]:


import statsmodels.api as sm


# In[169]:


sm.graphics.influence_plot(model1)


# In[170]:


startups_train = startup_train.drop([46])


# In[171]:


sm.graphics.plot_partregress_grid(model1)


# In[172]:


rsq_rd = smf.ols("RD~administration+Marketing+state_florida+state_newyork",data = startups_train).fit().rsquared
vif_rd = 1/(1-rsq_rd)


# In[173]:


rsq_rd = smf.ols("administration~RD+Marketing+state_florida+state_newyork",data = startups_train).fit().rsquared
vif_rd = 1/(1-rsq_rd)


# In[174]:


rsq_rd = smf.ols("Marketing~administration+RD+state_florida+state_newyork",data = startups_train).fit().rsquared
vif_rd = 1/(1-rsq_rd)


# In[175]:


rsq_sf = smf.ols("state_florida~RD+administration+Marketing+state_newyork",data = startups_train).fit().rsquared
vif_sf = 1/(1-rsq_sf)


# In[176]:


rsq_sf = smf.ols("state_newyork~RD+administration+Marketing+state_florida",data = startups_train).fit().rsquared
vif_sf = 1/(1-rsq_sf)


# In[177]:


model_mkdrop = smf.ols("profit~RD+Marketing+administration+state_florida+state_newyork",data = startups_train).fit()


# In[178]:


model_mkdrop.summary()


# In[179]:


print(model_mkdrop.conf_int(0.05))


# In[180]:


model_trainpred = model_mkdrop.predict(startups_train[["RD","administration","Marketing","state_florida","state_newyork"]])
errors_train = model_trainpred-startups_train['profit']


# In[181]:


np.sqrt(np.mean(errors_train*errors_train))


# In[182]:


plt.scatter(model_trainpred,startups_train['profit'],color = "blue");plt.xlabel("predicted values");plt.ylabel("actual value")


# In[183]:


plt.hist(model_mkdrop.resid_pearson)


# In[185]:


test_pred = model_mkdrop.predict(startup_test[['RD','administration','Marketing','state_florida','state_newyork']])


# In[186]:


errors_test = test_pred-startup_test['profit']


# In[187]:


np.sqrt(np.mean(errors_test*errors_test))

