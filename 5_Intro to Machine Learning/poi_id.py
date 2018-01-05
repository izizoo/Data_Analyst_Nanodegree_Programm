
# coding: utf-8

# In[154]:


import sys
import pickle
sys.path.append("/Users/Abdulaziz/DANP/All Projects/Project_6/tools")
import pandas as pd
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt
import scipy
#get_ipython().magic('matplotlib inline')


# In[155]:


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features
email_features_list=['from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages',
    ]

financial_features_list=['bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'total_payments',
    'total_stock_value',
]

features_list = ['poi']+email_features_list + financial_features_list 
### Load the dictionary containing the dataset
filename = '/Users/Abdulaziz/DANP/All Projects/Project_6/final_project_dataset.pkl'
with open(filename, "rb") as data_file:
    dataSet_dic = pickle.load(data_file)

#Dataset exploration
print ('==================================================Dataset Overview===================================================')
print('')
print(dataSet_dic)




# In[156]:


print ('Total number of People in the Dataset= ',len(dataSet_dic.keys()))

poi_Count=0
for name in dataSet_dic.keys():
    if dataSet_dic[name]['poi']==True:
        poi_Count+=1

print ('Number of Persons of Interest: {0}'.format(poi_Count))
print ('Number of Non-Person of Interest: {0}'.format(len(dataSet_dic.keys())-poi_Count))



# In[157]:


##Feature exploration
# Find missing data
all_features=dataSet_dic[name].keys()
print ('Total Features for everyone on the dataset:', len(all_features))

missing={}
for feature in all_features:
    missing[feature]=0

for person in dataSet_dic:
    records=0
    for feature in all_features:
        if dataSet_dic[person][feature]=='NaN':
            missing[feature]+=1
        else:
            records+=1

print ('Total Missing Values for each Feature:')

for feature in all_features:
    print (feature, missing[feature])


# In[158]:


### Task 2: Remove outliers

#function based on the multiple variables to visulize the data so outliers will be clear
def ShowOutlier(dataSet_dic, ax, ay):
    data = featureFormat(dataSet_dic, [ax,ay,'poi'])
    for point in data:
        x = point[0]
        y = point[1]
        poi=point[2]
        if poi:
            color='Red'
        else:
            color='Green'

        plt.scatter( x, y, color=color )
    plt.xlabel(ax)
    plt.ylabel(ay)
    plt.show()
ShowOutlier(dataSet_dic, 'from_poi_to_this_person','from_this_person_to_poi')
ShowOutlier(dataSet_dic, 'total_payments', 'total_stock_value')
ShowOutlier(dataSet_dic, 'from_messages','to_messages')
ShowOutlier(dataSet_dic, 'salary','bonus')


# In[159]:


##function to remove outliers
def remove_outliers(dataSet_dic, outliers):
    for outlier in outliers:
        dataSet_dic.pop(outlier, 0)
outliers =['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHARD EUGENE E']
remove_outliers(dataSet_dic, outliers)


# In[160]:


### Task 3: Create new feature(s) that will help identify POI
### Store to my_dataset for easy export below.
my_dataset = dataSet_dic

##Add new features to dataset


def getFraction( poi_messages, all_messages ):
    
    if all_messages =='NaN' or poi_messages == 'NaN':
        return 0.0
        
    return float(poi_messages)/float(all_messages)

submit_dict={}

for name in my_dataset:
    data_point = my_dataset[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = getFraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = getFraction( from_this_person_to_poi, from_messages )
    submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi}
    data_point["fraction_to_poi"] = fraction_to_poi


my_feature_list=features_list+['from_poi_to_this_person','to_messages','fraction_from_poi','from_this_person_to_poi',
                               'from_messages','fraction_to_poi']


# In[161]:


from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import cross_validation
from sklearn.svm import SVC


def getkbest(dataSet_dic, features_list, k):
    data=featureFormat(my_dataset, features_list)
    labels, features = targetFeatureSplit(data)
    selection=SelectKBest(k=k).fit(features,labels)
    scores=selection.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs=list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    selection_best = dict(sorted_pairs[:k])
    return selection_best

num=12
best_features = getkbest(my_dataset, my_feature_list, num)
print ('Selected features and their scores: ', best_features)
my_feature_list = ['poi'] + list(best_features)
print ("{0} selected features: {1}\n".format(len(my_feature_list) - 1, my_feature_list[1:]))


# In[162]:


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
from sklearn import preprocessing
scaler=preprocessing.MinMaxScaler()
features=scaler.fit_transform(features)


# In[163]:


from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit


# In[164]:


#DecisionTree
clf_d=Pipeline([
    ('standardscaler',StandardScaler()),
    ('pca',PCA()),
    ('clf_d',DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_leaf=2, min_samples_split=7, splitter='best',random_state=42))])


clf_p=Pipeline([
    ('standardscaler', StandardScaler()),
    ('classifier', LogisticRegression(penalty='l2', tol=0.001, C=0.0000001, random_state=42))])

from sklearn.cluster import KMeans
clf_k=Pipeline([
    ('standardscaler',StandardScaler()),
    ('pca',PCA()),
    ('clf_k',KMeans(n_clusters=2, random_state=42, tol=0.001))])

from sklearn.svm import SVC
clf_s=Pipeline([
    ('standardscaler',StandardScaler()),
    ('pca',PCA()),
    ('clf_s',SVC(kernel='rbf',C = 1000,random_state = 42))])


from sklearn.naive_bayes import GaussianNB
clf_g=Pipeline(steps=[
    ('standardscaler',StandardScaler()),
    ('pca',PCA()),
    ('clf_g',GaussianNB())])

from sklearn.ensemble import RandomForestClassifier
clf_rf =Pipeline( [
    ('standardscaler',StandardScaler()),
    ('pca',PCA()),
    ('clf_rf',RandomForestClassifier())]) 


# In[165]:


from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report
def evaluate(clf, features, labels, num=1000):
    print (clf)
    accuracy=[]
    precision=[]
    recall=[]
    for trial in range(num):
        features_train, features_test, labels_train, labels_test=                        cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)
        clf=clf.fit(features_train, labels_train)
        pred=clf.predict(features_test)
        accuracy.append(clf.score(features_test, labels_test))
        precision.append(precision_score(labels_test, pred))
        recall.append(recall_score(labels_test, pred))
    print ('precision: {}'.format(np.mean(precision)))
    print ('recall: {}'.format(np.mean(recall)))
    return np.mean(precision), np.mean(recall), confusion_matrix(labels_test, pred),classification_report(labels_test, pred) 


# In[166]:


import warnings
warnings.filterwarnings('ignore')

print ('KMeans: ', evaluate(clf_k, features, labels))


# In[167]:


print ('Gaussian: ', evaluate(clf_g, features, labels))


# In[168]:


print ('Linear Regression: ', evaluate(clf_p, features, labels))


# In[169]:


print ('Random Forest: ',evaluate(clf_rf, features, labels))


# In[170]:


print ('SVC: ', evaluate(clf_s, features, labels))


# In[171]:


print ('Decision Tree: ', evaluate(clf_d, features, labels))


# In[172]:


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# Example starting point. Try investigating other evaluation techniques!
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report
from sklearn.cross_validation import StratifiedShuffleSplit

skb = SelectKBest()
pca = PCA()
gnb = GaussianNB()
k_range = [6, 8, 10, 12]
PCA_range = [2, 3, 4, 5, 6]

steps = [('scaling',scaler), ('SKB', skb), ('pca',pca), ('algorithm', gnb)]
pipeline = Pipeline(steps)

parameters_gnb = {
        'SKB__k' : k_range,
        'pca__n_components' : PCA_range}

cv = StratifiedShuffleSplit(labels, n_iter=20, random_state = 42)

gs_gnb = GridSearchCV(pipeline, parameters_gnb, n_jobs = -1, cv=cv, scoring="f1")

gs_gnb.fit(features, labels)
clf = gs_gnb.best_estimator_


# In[173]:


print ('best estimator: ', gs_gnb.best_estimator_)
print ('best parameter: ', gs_gnb.best_params_)


# In[174]:


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
from tester import test_classifier

test_classifier(clf, my_dataset, features_list)
dump_classifier_and_data(clf, my_dataset, features_list)

