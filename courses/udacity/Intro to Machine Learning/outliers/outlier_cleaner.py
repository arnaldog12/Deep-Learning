#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    cleaned_data = [(a, n, (p-n)**2) for a,n,p in zip(ages, net_worths, predictions)]

    ### your code goes here
    last_10 = int(len(predictions)*0.1)
    cleaned_data = sorted(cleaned_data, key=lambda item:item[2])
    cleaned_data = cleaned_data[:-last_10]
    
    return cleaned_data