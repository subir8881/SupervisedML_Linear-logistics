#!/usr/bin/env python
# coding: utf-8

# In[80]:


import pandas as pd 
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf 
import matplotlib.pylab as plt
from sklearn import datasets, linear_model
from statsmodels.formula.api import ols
import statsmodels.api as sm


# In[35]:


Startup=pd.read_csv("50_Startups.csv")


# In[36]:


Startup.head(40)


# In[45]:


Startup.corr()


# In[46]:


sns.pairplot(Startup)


# In[66]:


sns.barplot(x='State',y='Profit', data=Startup, palette="Blues_d")


# In[47]:


Startup.columns


# In[67]:


sns.heatmap(Startup.corr(), annot=True)


# In[48]:


startup1 = Startup.rename({'R&D Spend': 'RD'}, axis= 1)


# In[49]:


startup2 = startup1.rename({'Marketing Spend': 'Marketing'}, axis= 1)


# In[69]:


startup2.head(5)


# In[70]:


startup3 = smf.ols('Profit~Administration+Marketing+State+RD',data=startup2).fit() 


# In[71]:


startup3.params


# In[73]:


startup3.summary()


# In[75]:


startupR=smf.ols('Profit~RD',data = startup2).fit()  
startupR.summary()


# In[76]:


startupM=smf.ols('Profit~Marketing',data = startup2).fit()  
startupM.summary()


# In[77]:


startupRM=smf.ols('Profit~RD+Marketing',data = startup2).fit()  
startupRM.summary()


# In[82]:


sm.graphics.influence_plot(startup3)


# In[84]:


startup4=startup2.drop(startup2.index[[48,46]],axis=0)


# In[86]:


startup_new = smf.ols('Profit~Administration+Marketing+State+RD',data = startup4).fit()    


# In[87]:


startup_new.params


# In[88]:


startup_new.summary()


# In[89]:


print(startup_new.conf_int(0.01))


# In[92]:


Profit_pred = startup_new.predict(startup4[['Profit','Administration','Marketing','RD','State']])


# In[93]:


Profit_pred


# In[94]:


startup4.head(5)


# In[98]:


rsq_RD = smf.ols('RD~Administration+Marketing+State',data=startup4).fit().rsquared  
vif_RD = 1/(1-rsq_RD)


# In[99]:


rsq_Administration = smf.ols('Administration~RD+Marketing+State',data=startup4).fit().rsquared  
vif_Administration = 1/(1-rsq_Administration)


# In[100]:


rsq_Marketing = smf.ols('Marketing~RD+Administration+State',data=startup4).fit().rsquared  
vif_Marketing = 1/(1-rsq_Marketing)


# In[101]:


rsq_State = smf.ols('State~RD+Administration+Marketing',data=startup4).fit().rsquared  
vif_State = 1/(1-rsq_State)


# In[103]:


STARTUP = {'Variables':['Administration','Marketing','RD'],'VIF':[vif_RD,vif_Administration,vif_Marketing]}
Vif_frame = pd.DataFrame(STARTUP)  
Vif_frame


# In[105]:


final_STARTUP= smf.ols('Profit~RD+Marketing',data = startup4).fit()
final_STARTUP.params
final_STARTUP.summary()


# In[106]:


Profit_pred = final_STARTUP.predict(startup4)


# In[110]:


sm.graphics.plot_partregress_grid(final_STARTUP)


# In[111]:


plt.scatter(startup4.Profit,Profit_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")


# In[112]:


plt.scatter(Profit_pred,final_STARTUP.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


# In[113]:


plt.hist(final_STARTUP.resid_pearson)


# In[114]:


import pylab          
import scipy.stats as st


# In[115]:


st.probplot(final_STARTUP.resid_pearson, dist="norm", plot=pylab)


# In[116]:


plt.scatter(Profit_pred,final_STARTUP.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


# In[120]:


from sklearn.model_selection import train_test_split
startup_train,startup_test  = train_test_split(startup4,test_size = 0.2)


# In[139]:


model_train = smf.ols("Profit~Marketing+State+RD",data=startup_train).fit()
print (model_train)


# In[137]:


train_pred = model_train.predict(startup_train)
print (test_pred )


# In[136]:


train_resid  = train_pred - cars_train.Profit
print (test_resid )


# In[135]:


train_rmse = np.sqrt(np.mean(train_resid*train_resid))
print (test_rmse )


# In[134]:


test_pred = model_train.predict(startup_test)
print (test_pred )


# In[133]:


test_resid  = test_pred - startup_test.Profit
print (test_resid )


# In[130]:


test_rmse = np.sqrt(np.mean(test_resid*test_resid))


# In[132]:


print (test_rmse)


# In[ ]:





# In[ ]:




