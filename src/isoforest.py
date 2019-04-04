#!/usr/bin/env python
# coding: utf-8

# # Outlier detection methods with PYOD on the wine quality dataset

# To see if it is possible to extract the best wines and worst wines through outlier detection

# In[1]:


import re
import numpy
import matplotlib.pyplot as plt
import plotly_express as px
import umap


# In[2]:


import pandas as pd
import pprint


# In[3]:


from pyod.models.iforest import IForest
from pyod.models.knn import KNN


# In[4]:


from lime.lime_tabular import LimeTabularExplainer
import shap


# ## Read in data from csv

# In[5]:


data = pd.read_csv("D:/NCS/repo/anomaly/data/winequality-white.csv", sep = ';')


# In[6]:


# Summary statistics for the dataset
data.describe()


# In[7]:


data_response = data['quality']
data.drop('quality', axis=1, inplace=True)


# In[8]:


type(data_response)


# Get count of response

# In[9]:


data_response.value_counts(sort=False).sort_index()


# ## Visualize data with UMAP

# In[10]:


reducer = umap.UMAP(n_neighbors=15, metric='euclidean', n_epochs=1000, random_state=100, verbose=True)


# In[11]:


embedding = reducer.fit_transform(data)


# In[12]:


embedding.shape


# In[13]:


UMAP_df = pd.DataFrame(embedding, columns = ['X1', 'X2'])


# In[14]:


UMAP_df.head(5)


# In[15]:


UMAP_df['response'] = data_response.astype(str)


# In[16]:


px.scatter(UMAP_df, x='X1', y='X2', color='response', width=600, height=400)


# There seem to be two major clusters of wine and a few outliers but the outliers do not seem to indicate the best or the worst wines

# ## Outlier detection with Isolation Forest

# In[17]:


# fit the model
clf = IForest(n_estimators=500,
              max_samples='auto',
              contamination=0.01,
              max_features=1.0, 
              bootstrap=False, 
              n_jobs=None, 
              random_state=42,
              verbose=0)

clf.fit(data)


# In[18]:


y_pred = clf.predict(data)


# In[19]:


pd.crosstab(y_pred, data_response, margins = True)


# ### Get predicted probabilities

# In[20]:


# Get original anomaly score
anomalies_score = clf.predict_proba(data, method ='unify')
anomalies_score_df = pd.DataFrame(anomalies_score, columns = ['non_outlier', 'outlier'])
anomalies_score_df.describe()


# In[21]:


px.histogram(anomalies_score_df, 
             x = 'outlier', 
             y = 'outlier', 
             width=600, 
             height=300)


# In[22]:


UMAP_df['prob'] = anomalies_score_df['outlier']


# In[23]:


heat_cols = px.colors.colorbrewer.RdYlBu[::-1]


# In[24]:


px.scatter(UMAP_df, x='X1', y='X2', color='prob', color_continuous_scale=heat_cols, width=600, height=400)


# The isolation forest seems to select outliers at the edges of the plot but not in the small clusters

# ### Use LIME and SHAP for outlier explanations

# In[25]:


anomalies_score_df.index[(anomalies_score_df['outlier'] > 0.9) & (anomalies_score_df['outlier'] < 0.99)].tolist()


# ### Lime explanations

# In[26]:


lime_explainer = LimeTabularExplainer(data.values, mode='classification', 
                                     class_names=['non-Outlier', 'Outlier'],
                                     feature_names=list(data.columns.values), 
                                     feature_selection = 'lasso_path',
                                     random_state=42, 
                                     discretize_continuous=True) 


# In[27]:


exp = lime_explainer.explain_instance(data.iloc[20,:].values, 
                                     lambda x: clf.predict_proba(x, method = 'unify'), 
                                     num_features=10)


# In[28]:


exp.as_pyplot_figure()


# In[29]:


exp.show_in_notebook(show_table=True, show_all=True)


# ### SHAP explanations

# In[30]:


shap.initjs()


# In[31]:


# Summarize data with median
data_summary = data.median().values.reshape((1,data.shape[1]))


# ### SHAP explanations for single observation

# In[32]:


import warnings
warnings.filterwarnings('ignore')

shap_explainer = shap.KernelExplainer(lambda x: clf.predict_proba(x, method = 'unify')[:,1], data_summary)


# In[33]:


x_test = data.iloc[17,:]

shap_values_single = shap_explainer.shap_values(x_test, nsamples=1000)
shap.force_plot(shap_explainer.expected_value, shap_values_single, x_test)


# ### SHAP explanations (variable importance)

# In[34]:


# Use train-test split to sample points for plotting
from sklearn.model_selection import train_test_split

_, data_sub, _, _= train_test_split(data, data_response, test_size=0.05, random_state=42, stratify=data_response)


# In[35]:


data_sub.shape


# In[36]:


shap_values = shap_explainer.shap_values(data_sub, nsamples=100, l1_reg = 'aic')


# In[37]:


shap.force_plot(shap_explainer.expected_value, shap_values, data_sub, figsize=(10, 2))


# There seem to be two types of outliers investigate particular values to check similarity

# In[38]:


import scipy

D = scipy.spatial.distance.pdist(shap_values, 'sqeuclidean')
clustOrder = scipy.cluster.hierarchy.leaves_list(scipy.cluster.hierarchy.complete(D))


# In[39]:


# calculate full dendrogram
#plt.figure(figsize=(20, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
scipy.cluster.hierarchy.dendrogram(
    Z = scipy.cluster.hierarchy.complete(D)
)
plt.show()


# In[40]:


cut_tree = scipy.cluster.hierarchy.cut_tree(scipy.cluster.hierarchy.complete(D), n_clusters=3)
cut_tree_df = pd.DataFrame(cut_tree)
cut_tree_df.iloc[:,0].value_counts()


# In[41]:


# Scale original data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

data_scaled = scaler.fit(data).transform(data)
data_scaled_df = pd.DataFrame(data_scaled, columns = data.columns)


# In[42]:


data_scaled_df.mean().round(2)


# In[43]:


data_scaled_df.std().round(2)


# In[44]:


# Index for subsample
grp1_index = numpy.where(cut_tree ==1)[0]
grp2_index = numpy.where(cut_tree ==2)[0]


# In[45]:


# Get index for original data
grp1_index = data_sub.iloc[numpy.where(cut_tree ==1)[0], :].index
grp2_index = data_sub.iloc[numpy.where(cut_tree ==2)[0], :].index


# In[46]:


grp1_index


# In[47]:


data_scaled_df.iloc[grp1_index,:].mean().round(2)


# In[48]:


data_scaled_df.iloc[grp2_index,:].mean().round(2)


# In[49]:


outlier_grps_df = pd.DataFrame({
    'vars':data_scaled_df.columns,
    'grp1':data_scaled_df.iloc[grp1_index,:].mean().tolist(),
    'grp2':data_scaled_df.iloc[grp2_index,:].mean().tolist()
})


# In[50]:


outlier_grps_df


# In[51]:


outlier_grps_df_long = pd.melt(outlier_grps_df,
                        id_vars=['vars'],
                        var_name='grp', 
                        value_name='values')


# In[52]:


px.histogram(outlier_grps_df_long, 
             x='vars', 
             y='values', 
             color='grp', 
             barmode='relative',
             histfunc='avg',
             orientation='v',
             labels={'values':'Standardized Values',
                    'vars':'Features'}, width=600, height=400)


# ### Global Variable importance using SHAP

# In[53]:


shap.summary_plot(shap_values, data_sub, plot_type='dot')


# In[54]:


shap.summary_plot(shap_values, data_sub, plot_type='bar')


# In[ ]:




