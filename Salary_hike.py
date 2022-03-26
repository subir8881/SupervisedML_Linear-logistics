#!/usr/bin/env python
# coding: utf-8

# In[102]:


import pandas as pd
import numpy as np
import sweetviz as sv
import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[30]:


pip install sweetviz


# In[5]:


SD=pd.read_csv("SalaryData.csv")


# In[7]:


SD.info


# In[9]:


SD.corr()


# In[13]:


SD.head()


# In[16]:


import seaborn as sns
sns.distplot(SD['YearsExperience'])


# In[17]:


import seaborn as sns
sns.distplot(SD['Salary'])


# In[37]:


sweet_report=sv.analyze(SD)
sweet_report.show_html('Salary_report.html')


# In[65]:


salary_hike=smf.ols("Salary~YearsExperience", data= SD). fit()


# In[66]:


sns.regplot(x="Salary", y="YearsExperience", data=SD);


# In[67]:


salary_hike.params


# In[68]:


print(salary_hike.tvalues, '\n',salary_hike.pvalues)


# In[69]:


salary_hike.summary()


# In[104]:


R-squared:0.957
Ajusted R-squared:0.955
Prob (F-statistic):1.14e-20

#Since R square and adj R square value is greater almost equal to 1 and probability is less than 0.05, we can consider salary over YearsExperience is the best model for the prediction.

