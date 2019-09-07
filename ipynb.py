#!/usr/bin/env python
# coding: utf-8

# # This is the G&M pp 94-97 sample code

# In[3]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import export_graphviz
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings


# In[4]:


from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
if Version(sklearn_version) < '0.18':
    from sklearn.grid_search import train_test_split
else:
    from sklearn.model_selection import train_test_split


# In[5]:


def versiontuple(v):
    return tuple(map(int, (v.split("."))))


# In[6]:


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')


# In[7]:


def gini(p):
    return (p)*(1 - (p)) + (1 - p)*(1 - (1-p))
def entropy(p):
    return - p*np.log2(p) - (1 - p)*np.log2((1 - p))
def error(p):
    return 1 - np.max([p, 1 - p])
x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]
fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err], 
                          ['Entropy', 'Entropy (scaled)', 'Gini Impurity', 'Misclassification Error'],
                          ['-', '-', '--', '-.'],
                          ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab,
    linestyle=ls, lw=2, color=c)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()


# In[8]:


mydata = pd.read_csv("test.csv")
mydata.set_index('rowindex',inplace=True)
mydata.drop('contract',axis=1,inplace=True)
mydata['squeeze']= mydata['squeeze'].map({True:1,False:0})
mydata.head()


# In[9]:


# Standardize the features
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
X=mydata[mydata.columns[:9]]
y=mydata["squeeze"]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=200)
print(X_train.shape, y_train.shape)


# In[11]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree.fit(X_train, y_train)


# In[18]:


X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))


# In[30]:


import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X_combined, y_combined,  clf=tree,  
                      filler_feature_values={1:0, 2: 0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0},
                      filler_feature_ranges={1:1, 2: 1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1})
plt.xlabel('feature 0')
plt.ylabel('feature 1')
plt.legend(loc= 'upper left')
plt.show()


# In[71]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(knn.score(X_train, y_train))
print(knn.score(X_test, y_test))


# In[104]:


from sklearn import metrics
k_range = range(1,26)
scores= []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))
plt.plot(scores)


# In[ ]:


# From the graph, we could see the bset choice of K is when K equals 4, 8, 13,23. 


# In[11]:


print("My name is Yuli Tang")
print("My NetID is: ytang56")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


# In[ ]:





# In[ ]:




