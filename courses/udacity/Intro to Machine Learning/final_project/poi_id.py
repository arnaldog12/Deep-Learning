#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', \
				'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', \
				'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', \
				'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', \
				'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL')

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
from math import isnan
for k,v in data_dict.items():
    from_pois_ratio =  float(v['from_poi_to_this_person'])/float(v['from_messages'])
    to_pois_ratio = float( v['from_this_person_to_poi'])/float(v['to_messages'])
    v['from_pos_ratio'] = 'NaN' if isnan(from_pois_ratio) else from_pois_ratio
    v['to_pois_ratio'] = 'NaN' if isnan(to_pois_ratio) else to_pois_ratio

my_dataset = data_dict

### Extract features and labels from dataset for local testing
features_list.append('from_pos_ratio')
features_list.append('to_pois_ratio')

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

print("       Number of features: ", len(features[0]))
print("        Number of samples: ", len(features))
print("    Number of POI samples: ", labels.count(1))
print("Number of Non-POI samples: ", labels.count(0))

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB

estimators = [('kbest', SelectKBest()), ('gaussian', GaussianNB())]
pipe = Pipeline(estimators)
params = dict(kbest__k=range(1,20))
clf = GridSearchCV(pipe, param_grid=params, scoring='f1')

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
clf.fit(features_train, labels_train)

print(clf.best_params_)

pred = clf.predict(features_test)
print(classification_report(labels_test, pred))
print(confusion_matrix(labels_test, pred))

from sklearn.preprocessing import MinMaxScaler

estimators = [('kbest', SelectKBest(k=6)), ('gaussian', GaussianNB())]
clf = Pipeline(estimators)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print(classification_report(labels_test, pred))
print(confusion_matrix(labels_test, pred))

kbest = estimators[0][1]
t = [(features_list[i+1], kbest.scores_[i]) for i in range(len(kbest.scores_))]
t.sort(key=lambda x:x[1], reverse=True)
print(t)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)