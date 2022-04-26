from scipy import stats
import numpy as np

def entropy(class_y):

    # Input:            
    #   class_y         : list of class labels (0's and 1's)

    _, counts = np.unique(class_y, return_counts=True)
    probs = counts/len(class_y)
    
    entropy = stats.entropy(probs, base = 2)
    
    return entropy


def partition_classes(X, y, split_attribute, split_val):

    # Inputs:
    #   X               : data containing all attributes
    #   y               : labels
    #   split_attribute : column index of the attribute to split on
    #   split_val       : either a numerical or categorical value to divide the split_attribute

    X_left = []
    X_right = []
    
    y_left = []
    y_right = []
      
    index = 0

    if (isinstance(X[0][split_attribute], str)):
        for row in X:
            if row[split_attribute] == split_val:
                X_left.append(row)
                y_left.append(y[index])
            else:
                X_right.append(row)
                y_right.append(y[index])
            index += 1
    else:
        for row in X:
            if row[split_attribute] <= split_val:
                X_left.append(row)
                y_left.append(y[index])
            else:
                X_right.append(row)
                y_right.append(y[index])
            index += 1

    return (X_left, X_right, y_left, y_right)

    
def information_gain(previous_y, current_y):
    
    # Inputs:
    #   previous_y: the distribution of original labels (0's and 1's)
    #   current_y:  the distribution of labels after splitting based on a particular
    #               split attribute and split value

    prev_entropy = entropy(previous_y)
    new_entropy = 0.0
    for row in current_y:
        prob = (len(row)/len(previous_y))
        new_entropy += prob * entropy(row)
    info_gain = prev_entropy - new_entropy
    return info_gain
    
    
def best_split(X, y):
    # Inputs:
    #   X       : Data containing all attributes
    #   y       : labels

    split_attribute = 0
    split_value = 0
    X_left, X_right, y_left, y_right = [], [], [], []
    old_gain = 0.0

    m = np.ceil(np.sqrt(len(X[0]))).astype(int)

    attributes = np.random.choice(len(X[0]), m, replace = False)

    for at in attributes:
        col = []

        for row in X:
            col.append(row[at])

        if (isinstance(X[0][at], str)):
            this_split = stats.mode(col)[0][0]
        else:
            this_split = np.mean(col)

        X_l, X_r, y_l, y_r = partition_classes(X, y, at, this_split)
        new_gain = information_gain(y, [y_l, y_r])
        if new_gain > old_gain:
            X_left, X_right, y_left, y_right = X_l, X_r, y_l, y_r
            new_gain = old_gain
            split_attribute = at
            split_value = this_split
        old_gain = new_gain
    if ((X_left == []) or (X_right == []) or (y_left == []) or (y_right == [])):
        split_attribute = 'None'

    return (split_attribute, split_value, X_left, X_right, y_left, y_right)
                

    

    
