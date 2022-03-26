#!/usr/bin/env python
# coding: utf-8

# In[86]:


import pandas as pd
import numpy as np
from statsmodels.graphics.regressionplots import influence_plot
import matplotlib.pyplot as plt
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[87]:


toyota=pd.read_csv('ToyotaCorolla2.txt')


# In[88]:


toyota.info()


# In[89]:


toyota.head()


# In[90]:


t1=toyota.iloc[:,[2,3,6,8,12,13,15,16,17]]


# In[91]:


t2=t1.rename({'Age_08_04':'Age'}, axis=1)


# In[92]:


t2.shape


# In[93]:


t2.isna().sum()


# In[94]:


t2.describe()


# In[95]:


t2.corr()


# In[96]:


import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(t2)


# In[97]:


plt.boxplot(t2["Price"])


# In[98]:


plt.boxplot(t2["Age"])
plt.boxplot(t2["HP"])
plt.boxplot(t2["Quarterly_Tax"])


# In[99]:


import statsmodels.formula.api as smf 
model1 = smf.ols('Price~+Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=t2).fit()
model1.summary()


# In[100]:



m1_cc = smf.ols("Price~cc",data= t2).fit()
m1_cc.summary()


# In[101]:



m1_Doors = smf.ols("Price~Doors",data= t2).fit()
m1_Doors.summary()


# In[102]:



m1_cD = smf.ols("Price~cc+Doors",data= t2).fit()
m1_cD.summary()


# In[103]:


rsq_age=smf.ols('Age~KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data=t2).fit().rsquared
vif_age=1/(1-rsq_age)

rsq_km=smf.ols('KM~Age+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data=t2).fit().rsquared
vif_age=1/(1-rsq_km)

rsq_hp=smf.ols('HP~KM+Age+cc+Doors+Gears+Quarterly_Tax+Weight', data=t2).fit().rsquared
vif_age=1/(1-rsq_hp)

rsq_cc=smf.ols('cc~KM+HP+Age+Doors+Gears+Quarterly_Tax+Weight', data=t2).fit().rsquared
vif_age=1/(1-rsq_cc)

rsq_doors=smf.ols('Doors~KM+HP+cc+Age+Gears+Quarterly_Tax+Weight', data=t2).fit().rsquared
vif_age=1/(1-rsq_doors)

rsq_gears=smf.ols('Gears~KM+HP+cc+Doors+Age+Quarterly_Tax+Weight', data=t2).fit().rsquared
vif_age=1/(1-rsq_gears)

rsq_qt=smf.ols('Quarterly_Tax~KM+HP+cc+Doors+Gears+Age+Weight', data=t2).fit().rsquared
vif_age=1/(1-rsq_qt)

rsq_weight=smf.ols('Weight~KM+HP+cc+Doors+Gears+Quarterly_Tax+Age', data=t2).fit().rsquared
vif_age=1/(1-rsq_weight)


# In[104]:


D1={'Variables' : ['Age', 'KM', 'HP', 'cc', 'Doors', 'Gears', 'Quarterly_Tax', 'Weight'],
   'VIF': [rsq_age,rsq_km,rsq_hp,rsq_cc,rsq_doors,rsq_gears,rsq_qt,rsq_weight]}
Vif_frame=pd.DataFrame(D1)
Vif_frame


# In[105]:




res=model1.resid
res


# In[106]:



import statsmodels.api as sm
qqplot=sm.qqplot(res, line='q')
plt.title("Test for Normality of Residuals(Q-Q plot)")
plt.show


# In[107]:




list(np.where(model1.resid>6000))


# In[108]:


def get_standardized_values(vals) :
    return (vals-vals.mean())/vals.std()


# In[109]:


def get_standardized_values(vals) :
    return (vals-vals.mean())/vals.std()


# In[110]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model1, "Age", fig=fig)
plt.show()


# In[111]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model1, "KM", fig=fig)
plt.show()


# In[112]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model1, "HP", fig=fig)
plt.show()


# In[113]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model1, "cc", fig=fig)
plt.show()


# In[114]:



fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model1, "Doors", fig=fig)
plt.show()


# In[115]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model1, "Gears", fig=fig)
plt.show()


# In[116]:



fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model1, "Quarterly_Tax", fig=fig)
plt.show()


# In[117]:



fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model1, "Weight", fig=fig)
plt.show()


# In[118]:


model_influence = model1.get_influence()
(c, _) = model_influence.cooks_distance


# In[119]:


c


# In[120]:


fig = plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(toyota)), np.round(c,3))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()


# In[121]:


np.argmax(c), np.max(c)


# In[122]:



from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model1)
plt.show()


# In[123]:


k = toyota.shape[1]
n = toyota.shape[0]
leverage_cutoff = 3*((k+1)/n)
leverage_cutoff


# In[124]:


toyota[toyota.index.isin([80,221,960])]
toyota.head()


# In[125]:


T_new = toyota.drop(toyota.index[[80,221,960]], axis=0).reset_index()
T_new


# In[126]:


T_N2=T_new.drop(['index'], axis=1)
T_N2


# In[127]:


sm.graphics.plot_partregress_grid(model1)


# In[128]:


final_ml_wd = smf.ols('Price~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight', data=T_N2).fit()


# In[129]:


(final_ml_wd.rsquared, final_ml_wd.aic)


# In[130]:


final_ml_wg = smf.ols('Price~Age_08_04+KM+HP+cc+Quarterly_Tax+Weight', data=T_N2).fit()

(final_ml_wg.rsquared, final_ml_wg.aic)


# In[131]:


model_influence_wd = final_ml_wd.get_influence()
(c_wd, _) = model_influence_wd.cooks_distance


# In[132]:


fig = plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(T_N2)), np.round(c_wd,3))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()


# In[133]:



np.argmax(c_wd), np.max(c_wd)


# In[134]:


T_3 = T_N2.drop(T_N2.index[[599]], axis=0).reset_index()
T_3


# In[135]:


T_4 = T_3.drop(['index'], axis=1)
T_4


# In[136]:


final_ml_wd=smf.ols('Price~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight', data=T_4).fit()
model_influence_wd = final_ml_wd.get_influence()
(c_vol, _) = model_influence_wd.cooks_distance


# In[137]:


fig = plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(T_4)), np.round(c_wd,3))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()


# In[138]:


np.argmax(c_wd), np.max(c_wd)


# In[139]:


(final_ml_wd.rsquared, final_ml_wd.aic)


# In[140]:


sm.graphics.plot_partregress_grid(final_ml_wd)


# In[141]:


new_D=pd.DataFrame({'Age_08_04':78 , 'KM':20600 , 'HP':120 , 'cc':2000 , 'Gears':6 , 'Quarterly_Tax':234 , 'Weight':1300 }, index=[1])


# In[142]:


new_D


# In[143]:


final_ml_wd.predict(new_D)


# In[144]:


pred_y = final_ml_wd.predict(toyota)


# In[145]:


pred_y

