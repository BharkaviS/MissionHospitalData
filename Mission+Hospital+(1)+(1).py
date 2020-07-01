
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_excel("Mission Hospital-Case Data.xlsx")


# In[3]:


data=pd.ExcelFile('Mission Hospital-Case Data.xlsx')
print(data.sheet_names)


# In[4]:


df1=data.parse("MH-Modified Data")


# df1.head()

# In[5]:


print(df1.head())


# In[6]:


df.shape


# df.columns

# In[7]:


print(df1.columns)


# In[9]:


df1=df1.drop(['GENDER','MARITAL STATUS','KEY COMPLAINTS -CODE','PAST MEDICAL HISTORY CODE','MODE OF ARRIVAL','STATE AT THE TIME OF ARRIVAL','IMPLANT USED (Y/N)'],axis=1)


# In[10]:


print(df1.columns)

cor=df1.corr()
# In[8]:


print(df1.corr())


# In[12]:


x=df1.iloc[:,[1,5,17,23,]]


# In[13]:


x


# In[15]:


sns.heatmap(x.corr(),annot=True)
plt.show()


# In[17]:


sns.barplot(x=df1['CAD-DVD'],y=df1['AGE'])
plt.show()


# In[19]:


sns.barplot(x=df1['CAD-DVD'],y=df1['BODY WEIGHT'])
plt.show()


# In[43]:


sns.barplot(x=df1['CAD-DVD'],y=df1['Diabetes1'])


# In[20]:


X=df1.iloc[:,[1,17,23,25]]


# In[21]:


X

sns.regplot(x=df1['AGE'],y=df1['CAD-DVD'])
# In[183]:


sns.regplot(x=df1['AGE'],y=df1['BODY WEIGHT'])
plt.ylim(0,)


# In[184]:


y=df1.iloc[:,[5]]


# In[185]:


# Splitting the dataset into Training Set and Test Set 
from sklearn.cross_validation import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 0) 


# In[186]:


# Feature Scaling 
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler() 
X_train = sc_X.fit_transform(X_train) 
X_test = sc_X.transform(X_test) 


# In[187]:


# Fitting Logistic Regression to Training Dataset 
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state = 0) 
classifier.fit(X_train, y_train) 



# In[194]:


# Predicting the Test Set Results 


y_pred = classifier.predict(X_test) 
y_pred


# In[235]:


x_test=[[20,38,0,1]]
y_pred = classifier.predict(x_test) 
y_pred


# In[222]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3, random_state = 0)


# In[200]:


from sklearn.linear_model import 
lr = LinearRegression()
lr.fit(X_train, y_train)


# In[205]:


x_new=[[60,70,1,1]]
y_pred = lr.predict(x_new)
y_pred


# In[232]:


from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="gini")

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
X_new=[[20,38,0,1]]
#Predict the response for test dataset
y_pred = clf.predict(X_new)


# In[233]:


print(y_pred)


# In[234]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[236]:


from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# In[237]:


y_pred


# In[238]:


x_new=[[40,65,0,0]]
y_pred=clf.predict(x_new)
y_pred


# In[239]:


x_new=[[40,65,0,1]]
y_pred=clf.predict(x_new)
y_pred


# In[240]:


x_new=[[40,80,1,1]]
y_pred=clf.predict(x_new)
y_pred

