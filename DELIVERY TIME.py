#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
import numpy as np
import sweetviz as sv
import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[4]:


DT=pd.read_csv("Delivery_time.csv")


# In[5]:


DT.describe()


# In[6]:


DT.corr()


# In[9]:


import seaborn as sns
sns.distplot(DT['Delivery Time'])


# In[10]:


import seaborn as sns
sns.distplot(DT['Sorting Time'])


# In[19]:


DT.info


# In[36]:


DT_cleaned1=DT.rename({'Sorting Time': 'Sorting'}, axis= 1)


# In[37]:


print (DT_cleaned1)


# In[38]:


DT_cleaned2=DT_cleaned1.rename({'Delivery Time': 'Deliver'}, axis= 1)


# In[39]:


print (DT_cleaned2)


# In[43]:


DT_1=smf.ols("Deliver~Sorting", data= DT_cleaned2). fit()


# In[45]:


sns.regplot(x="Deliver", y="Sorting", data=DT_cleaned2);


# In[65]:


sweet_report=sv.analyze(DT_cleaned2)
sweet_report.show_html('DT_report.html')


# In[62]:


DT_1.params


# In[63]:


print(DT_1.tvalues, '\n',DT_1.pvalues)


# In[64]:


DT_1.summary()


# In[ ]:


#Since the P-value is less than 0.05. So X varibale is significance and also Multiple R-Square value is 0.6823. That’s mean this model will predict the output 68.23% time correct

