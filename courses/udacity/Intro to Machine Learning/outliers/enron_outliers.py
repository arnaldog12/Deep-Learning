#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
del(data_dict['TOTAL'])
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

### your code below
for point in data:
	salary = point[0]
	bonus = point[1]
	matplotlib.pyplot.scatter(salary, bonus)

data_clean = {k:v for k,v in data_dict.items() if v['salary'] != 'NaN' and v['bonus'] != 'NaN'}
bandits = [k for k in data_clean if data_clean[k]['salary'] > 1000000 and data_clean[k]['bonus'] > 5000000]
print(bandits)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()