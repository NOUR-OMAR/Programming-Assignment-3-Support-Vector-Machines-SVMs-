#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import nessesry libraries 
import pandas as pd 
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import LinearSVC, SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import warnings
warnings.filterwarnings('ignore') 


# In[2]:


#read dataset file
data = pd.read_csv("SVMdataset.csv") 
data.head()


# In[3]:


features=['x1','x2'] # Features
x=data[features]
y=data['y']
x=x.values.reshape(-2,2)


# In[4]:


#splitting data 
from  sklearn.model_selection  import train_test_split
X_train,X,y_train,Y=train_test_split(x,y,train_size=0.6,test_size=0.4,random_state=42)
X_cv,X_test,y_cv,y_test= train_test_split(X,Y,test_size = 0.50,train_size =0.50,random_state=42)


# In[5]:


#plotting train dataset 
from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit_transform(X_train)
plt.figure(figsize=(10, 6))
plt.scatter(X_train [y_train== 0][:, 0], X_train [y_train== 0][:, 1],marker='o', color='r', label='-ve')
plt.scatter(X_train [y_train== 1][:, 0], X_train [y_train== 1][:, 1],marker='x', color='b', label='+ve')
plt.legend();
plt.title('train SVM dataset')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


# In[6]:


#plotting test dataset 

X_test = StandardScaler().fit_transform(X_test )
plt.figure(figsize=(10, 6))
plt.scatter(X_test [y_test== 0][:, 0], X_test [y_test== 0][:, 1],marker='o', color='r', label='-ve')
plt.scatter(X_test [y_test== 1][:, 0], X_test [y_test== 1][:, 1],marker='x', color='b', label='+ve')
plt.legend();
plt.title('test SVM dataset')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


# In[7]:


#plotting validation dataset 

X_cv= StandardScaler().fit_transform(X_cv)
plt.figure(figsize=(10, 6))
plt.scatter(X_cv[y_cv== 0][:, 0],X_cv [y_cv== 0][:, 1],marker='o', color='r', label='-ve')
plt.scatter(X_cv [y_cv== 1][:, 0],X_cv [y_cv== 1][:, 1],marker='x', color='b', label='+ve')
plt.legend();
plt.title('validation SVM dataset')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


# In[8]:


print('classifier with linear kernel.')


# In[9]:


#For the classifier with linear kernel i create a for loop to try a range of C values to find
#the C value that gives the best F1 score using the validation data.


c_values_l=[]

for i in range(1,100,2):
    i=i*0.001
    c_values_l.append(i)
max_f1_score = float('-inf')
best_c = None
for c in c_values_l:

    linear_clf = svm.SVC(kernel='linear',C=c)
    linear_clf.fit(X_train, y_train)
    y_pred = linear_clf.predict(X_cv)
    current_f1_score = metrics.f1_score(y_cv, y_pred)

    if current_f1_score > max_f1_score:
        max_f1_score = current_f1_score
        best_c = c


print('best c: ',best_c)


# In[10]:


#Print the values of accuracy, prescion, recall, and F1 score for the the validation datasets

print('classification report for validation SVM dataset')
linear_clf = svm.SVC(kernel='linear',C=best_c)
linear_clf.fit(X_train, y_train)
y_pred_l = linear_clf.predict(X_cv)
print(confusion_matrix(y_cv,y_pred_l))
print(classification_report(y_cv,y_pred_l))
print('************************')
cnf_matrix_l = metrics.confusion_matrix(y_cv, y_pred_l)
tn, fp, fn, tp = metrics.confusion_matrix(y_cv, y_pred_l).ravel()
print("confusion_matrix:",cnf_matrix_l)
print("tn: ",tn," ",",fp: ", fp," ",",fn: ", fn," " ,",tp: ", tp)
print('************************')
print('classification report for validation SVM dataset')
acc=metrics.accuracy_score(y_cv, y_pred_l)
print("Accuracy:",acc)
pre=metrics.precision_score(y_cv, y_pred_l)
print("Precision:",pre)
rec=metrics.recall_score(y_cv, y_pred_l)
print("Recall:",rec)
fscore=metrics.f1_score(y_cv, y_pred_l)
print("f1 score:",fscore)
auc=metrics.roc_auc_score(y_cv, y_pred_l)
print("auc roc:",auc)


# In[11]:


fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cnf_matrix_l)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(s[i][j])+" = "+str(cnf_matrix_l[i][j]), ha='center', va='center', color='black')
plt.title('confusion_matrix validation SVM dataset')
plt.show()


# In[12]:


print('confusion_matrix validation SVM dataset')
plot_confusion_matrix(linear_clf, X_cv, y_cv)


# In[13]:


#Print linear svc decision boundary for validation dataset

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

linear_clf = svm.SVC(kernel='linear',C=best_c)
linear_clf.fit(X_train, y_train)

fig, ax = plt.subplots()

xx, yy = make_meshgrid( X_cv[:, 0], X_cv[:, 1])

plot_contours(ax, linear_clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)

plt.scatter(X_cv[y_cv== 0][:, 0],X_cv [y_cv== 0][:, 1],marker='o', color='r', label='-ve')
plt.scatter(X_cv [y_cv== 1][:, 0],X_cv [y_cv== 1][:, 1],marker='x', color='b', label='+ve')

ax.set_ylabel('y_cv')
ax.set_xlabel('X_cv')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('decision boundary for validation SVM dataset')
ax.legend()
plt.show()


# In[14]:


#Print linearconfusion_matrix train SVM dataset

print('confusion_matrix train SVM dataset')
plot_confusion_matrix(linear_clf, X_train, y_train)


# In[15]:


#Print the values of accuracy, prescion, recall, and F1 score for the the train datasets

print('classification report for train SVM dataset')
linear_clf = svm.SVC(kernel='linear',C=best_c)
linear_clf.fit(X_train, y_train)
y_pred_l = linear_clf.predict(X_train)
print(confusion_matrix(y_train,y_pred_l))
print(classification_report(y_train,y_pred_l))
print('************************')
cnf_matrix_l = metrics.confusion_matrix(y_train, y_pred_l)
tn, fp, fn, tp = metrics.confusion_matrix(y_train, y_pred_l).ravel()
print("confusion_matrix:",cnf_matrix_l)
print("tn: ",tn," ",",fp: ", fp," ",",fn: ", fn," " ,",tp: ", tp)
print('************************')
print('classification report for train SVM dataset')
acc=metrics.accuracy_score(y_train, y_pred_l)
print("Accuracy:",acc)
pre=metrics.precision_score(y_train, y_pred_l)
print("Precision:",pre)
rec=metrics.recall_score(y_train, y_pred_l)
print("Recall:",rec)
fscore=metrics.f1_score(y_train, y_pred_l)
print("f1 score:",fscore)
auc=metrics.roc_auc_score(y_train, y_pred_l)
print("auc roc:",auc)


# In[16]:


#Print linear svc decision boundary for training dataset

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

linear_clf = svm.SVC(kernel='linear',C=best_c)
linear_clf.fit(X_train, y_train)

fig, ax = plt.subplots()

xx, yy = make_meshgrid( X_train[:, 0], X_train[:, 1])

plot_contours(ax, linear_clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)

plt.scatter(X_train[y_train== 0][:, 0],X_train [y_train== 0][:, 1],marker='o', color='r', label='-ve')
plt.scatter(X_train [y_train== 1][:, 0],X_train [y_train== 1][:, 1],marker='x', color='b', label='+ve')

ax.set_ylabel('y_train')
ax.set_xlabel('x_train')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('decision boundary for train SVM dataset')
ax.legend()
plt.show()


# In[17]:


print('confusion_matrix test SVM dataset')

plot_confusion_matrix(linear_clf, X_test, y_test)


# In[18]:


#Print the values of accuracy, prescion, recall, and F1 score for the the test datasets

print('classification report for test SVM dataset')
linear_clf = svm.SVC(kernel='linear',C=best_c)
linear_clf.fit(X_train, y_train)
y_pred_l = linear_clf.predict(X_test)
print(confusion_matrix(y_test,y_pred_l))
print(classification_report(y_test,y_pred_l))
print('************************')
cnf_matrix_l = metrics.confusion_matrix(y_test,y_pred_l)
tn, fp, fn, tp = metrics.confusion_matrix(y_test,y_pred_l).ravel()
print("confusion_matrix:",cnf_matrix_l)
print("tn: ",tn," ",",fp: ", fp," ",",fn: ", fn," " ,",tp: ", tp)
print('************************')
print('classification report for test SVM dataset')
acc=metrics.accuracy_score(y_test,y_pred_l)
print("Accuracy:",acc)
pre=metrics.precision_score(y_test,y_pred_l)
print("Precision:",pre)
rec=metrics.recall_score(y_test,y_pred_l)
print("Recall:",rec)
fscore=metrics.f1_score(y_test,y_pred_l)
print("f1 score:",fscore)
auc=metrics.roc_auc_score(y_test,y_pred_l)
print("auc roc:",auc)


# In[19]:


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

linear_clf = svm.SVC(kernel='linear',C=best_c)
linear_clf.fit(X_train, y_train)

fig, ax = plt.subplots()
# title for the plots

xx, yy = make_meshgrid( X_test[:, 0], X_test[:, 1])

plot_contours(ax, linear_clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)

plt.scatter(X_test[y_test== 0][:, 0],X_test [y_test== 0][:, 1],marker='o', color='r', label='-ve')
plt.scatter(X_test [y_test== 1][:, 0],X_test [y_test== 1][:, 1],marker='x', color='b', label='+ve')

ax.set_ylabel('y_test')
ax.set_xlabel('x_test')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('decision boundary for test SVM dataset')
ax.legend()
plt.show()


# In[ ]:





# In[20]:


print('*****************************')
print('classifier with gaussian kernel.')


# In[22]:


#For the classifier with gaussian kernel i create a for loop to try a range of C values and gamma values to find
#the C value and gamma value that gives the best F1 score using the validation data.

c_values_rbf =[50,60,70,80,90,100]
gamma_values_rbf= [50,60,70,80,90,100]
  
max_f1_score = float('-inf')
best_c = None
best_g= None


for c in c_values_rbf:
    for g in gamma_values_rbf:
        rbf_clf = svm.SVC(kernel='rbf',C=c, gamma=g)
        rbf_clf.fit(X_train, y_train)
        y_pred_rbf = rbf_clf.predict(X_cv)
        current_f1_score = metrics.f1_score(y_cv, y_pred)
        if current_f1_score > max_f1_score:
            max_f1_score = current_f1_score
            best_c = c
            best_g=g
            
print('best c: ',best_c)
print('best g: ',best_g)


# In[23]:


print('classification report for validation SVM dataset')
rbf_clf = svm.SVC(kernel='rbf',C=best_c,gamma=best_g)
rbf_clf.fit(X_train, y_train)
y_pred_rbf= rbf_clf.predict(X_cv)
print(confusion_matrix(y_cv,y_pred_rbf))
print(classification_report(y_cv,y_pred_rbf))        

print('************************')
cnf_matrix_rbf = metrics.confusion_matrix(y_cv,y_pred_rbf)
tn, fp, fn, tp = metrics.confusion_matrix(y_cv,y_pred_rbf).ravel()
print("confusion_matrix:",cnf_matrix_rbf)
print("tn: ",tn," ",",fp: ", fp," ",",fn: ", fn," " ,",tp: ", tp)
print('************************')
print('classification report for validation SVM dataset')

acc=metrics.accuracy_score(y_cv,y_pred_rbf)
print("Accuracy:",acc)
pre=metrics.precision_score(y_cv,y_pred_rbf)
print("Precision:",pre)
rec=metrics.recall_score(y_cv,y_pred_rbf)
print("Recall:",rec)
fscore=metrics.f1_score(y_cv,y_pred_rbf, average='weighted')
print("f1 score:",fscore)
auc=metrics.roc_auc_score(y_cv,y_pred_rbf)
print("auc roc:",auc)


# In[24]:


fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cnf_matrix_rbf)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(s[i][j])+" = "+str(cnf_matrix_rbf[i][j]), ha='center', va='center', color='black')
print('confusion_matrix validation SVM dataset')
plt.show()


# In[25]:


print('confusion_matrix validation SVM dataset')
plot_confusion_matrix(rbf_clf, X_cv, y_cv)       


# In[26]:


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

rbf_clf = svm.SVC(kernel='rbf',C=best_c,gamma=best_g)
rbf_clf.fit(X_train, y_train)

fig, ax = plt.subplots()

xx, yy = make_meshgrid( X_cv[:, 0], X_cv[:, 1])

plot_contours(ax, rbf_clf, xx, yy, cmap=plt.cm.Paired, alpha=0.8)

plt.scatter(X_cv[y_cv== 0][:, 0],X_cv [y_cv== 0][:, 1],marker='o', color='r', label='-ve')
plt.scatter(X_cv [y_cv== 1][:, 0],X_cv [y_cv== 1][:, 1],marker='x', color='b', label='+ve')
# ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('y_cv')
ax.set_xlabel('X_cv')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('decision boundary for validation SVM dataset')
ax.legend()
plt.show()


# In[27]:


print('classification report for test SVM dataset')
rbf_clf = svm.SVC(kernel='rbf',C=best_c,gamma=best_g)
rbf_clf.fit(X_train, y_train)
y_pred_rbf = rbf_clf.predict(X_test)
print(confusion_matrix(y_test,y_pred_rbf))
print(classification_report(y_test,y_pred_rbf))        

print('************************')
cnf_matrix_rbf = metrics.confusion_matrix(y_test,y_pred_rbf)
tn, fp, fn, tp = metrics.confusion_matrix(y_test,y_pred_rbf).ravel()
print("confusion_matrix:",cnf_matrix_rbf)
print("tn: ",tn," ",",fp: ", fp," ",",fn: ", fn," " ,",tp: ", tp)
print('************************')
print('classification report for test SVM dataset')
acc=metrics.accuracy_score(y_test,y_pred_rbf)
print("Accuracy:",acc)
pre=metrics.precision_score(y_test,y_pred_rbf)
print("Precision:",pre)
rec=metrics.recall_score(y_test,y_pred_rbf)
print("Recall:",rec)
fscore=metrics.f1_score(y_test,y_pred_rbf)
print("f1 score:",fscore)
auc=metrics.roc_auc_score(y_test,y_pred_rbf)
print("auc roc:",auc)


# In[28]:


print('confusion_matrix test SVM dataset')
plot_confusion_matrix(rbf_clf, X_test, y_test)       


# In[29]:


fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cnf_matrix_rbf)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(s[i][j])+" = "+str(cnf_matrix_rbf[i][j]), ha='center', va='center', color='black')
print('confusion_matrix test SVM dataset')
plt.show()


# In[30]:


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

rbf_clf = svm.SVC(kernel='rbf',C=best_c,gamma=best_g)
rbf_clf.fit(X_train, y_train)

fig, ax = plt.subplots()

xx, yy = make_meshgrid( X_test[:, 0], X_test[:, 1])

plot_contours(ax, rbf_clf, xx, yy, cmap=plt.cm.Paired, alpha=0.8)

plt.scatter(X_test[y_test== 0][:, 0],X_test [y_test== 0][:, 1],marker='o', color='r', label='-ve')
plt.scatter(X_test [y_test== 1][:, 0],X_test [y_test== 1][:, 1],marker='x', color='b', label='+ve')
# ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('y_test')
ax.set_xlabel('X_test')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('decision boundary for test SVM dataset')
ax.legend()
plt.show()


# In[31]:


print('classification report for train SVM dataset')
rbf_clf = svm.SVC(kernel='rbf',C=best_c,gamma=best_g)
rbf_clf.fit(X_train, y_train)
y_pred_rbf= rbf_clf.predict(X_train)
print(confusion_matrix(y_train,y_pred_rbf))
print(classification_report(y_train,y_pred_rbf))        

print('************************')
cnf_matrix_rbf = metrics.confusion_matrix(y_train,y_pred_rbf)
tn, fp, fn, tp = metrics.confusion_matrix(y_train,y_pred_rbf).ravel()
print("confusion_matrix:",cnf_matrix_rbf)
print("tn: ",tn," ",",fp: ", fp," ",",fn: ", fn," " ,",tp: ", tp)
print('************************')
print('classification report for train SVM dataset')
acc=metrics.accuracy_score(y_train,y_pred_rbf)
print("Accuracy:",acc)
pre=metrics.precision_score(y_train,y_pred_rbf)
print("Precision:",pre)
rec=metrics.recall_score(y_train,y_pred_rbf)
print("Recall:",rec)
fscore=metrics.f1_score(y_train,y_pred_rbf)
print("f1 score:",fscore)
auc=metrics.roc_auc_score(y_train,y_pred_rbf)
print("auc roc:",auc)


# In[32]:


print('confusion_matrix train SVM dataset')
plot_confusion_matrix(rbf_clf, X_train, y_train)      


# In[33]:


fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cnf_matrix_rbf)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(s[i][j])+" = "+str(cnf_matrix_rbf[i][j]), ha='center', va='center', color='black')
print('confusion_matrix train SVM dataset')
plt.show()


# In[34]:


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

rbf_clf = svm.SVC(kernel='rbf',C=best_c,gamma=best_g)
rbf_clf.fit(X_train, y_train)

fig, ax = plt.subplots()

xx, yy = make_meshgrid( X_train[:, 0], X_train[:, 1])

plot_contours(ax, rbf_clf, xx, yy, cmap=plt.cm.Paired, alpha=0.8)

plt.scatter(X_train[y_train== 0][:, 0],X_train [y_train== 0][:, 1],marker='o', color='r', label='-ve')
plt.scatter(X_train [y_train== 1][:, 0],X_train [y_train== 1][:, 1],marker='x', color='b', label='+ve')
# ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('y_train')
ax.set_xlabel('X_train')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('decision boundary for train SVM dataset')
ax.legend()
plt.show()


# In[ ]:




