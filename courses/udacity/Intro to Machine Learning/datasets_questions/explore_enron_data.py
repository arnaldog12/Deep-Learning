#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print(len(enron_data))
print(len(enron_data['METTS MARK']))

pois = {k:v for k,v in enron_data.items() if v['poi']}
print(len(pois))

print(enron_data['PRENTICE JAMES']['total_stock_value'])
print(enron_data['COLWELL WESLEY']['from_this_person_to_poi'])
print(enron_data['SKILLING JEFFREY K']['exercised_stock_options'])

print(enron_data['LAY KENNETH L']['total_payments'])
print(enron_data['SKILLING JEFFREY K']['total_payments'])
print(enron_data['FASTOW ANDREW S']['total_payments'])

quant_salary = {k:v for k,v in enron_data.items() if v['salary'] != 'NaN'}
email_address = {k:v for k,v in enron_data.items() if v['email_address'] != 'NaN'}
print(len(quant_salary), len(email_address))

nan_total_payments = {k:v for k,v in enron_data.items() if v['total_payments'] == 'NaN'}
len_nan_tp = float(len(nan_total_payments))
print(len_nan_tp, len_nan_tp/len(enron_data))

poi_nan_tp = {k:v for k,v in nan_total_payments.items() if v['poi']}
len_poi_nan_tp = float(len(poi_nan_tp))
print(len_poi_nan_tp, len_poi_nan_tp/len(enron_data))

print(len(enron_data)+10, len_nan_tp+10)
print(len(pois)+10, len_poi_nan_tp+10)