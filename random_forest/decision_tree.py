from util import best_split, stats
import numpy as np 
import ast

class DecisionTree(object):
    def __init__(self, max_depth):
        self.tree = {}
        self.max_depth = max_depth
    	
    def learn(self, X, y, par_node = {}, depth=0):
        if (depth > self.max_depth) or (len(np.unique(y)) == 1):
            self.tree = stats.mode(y)[0][0]
        else:
            split_attribute, split_value, X_left, X_right, y_left, y_right = best_split(X, y)
            if (isinstance(split_attribute, str)):
                self.tree = stats.mode(y)[0][0]
            else:
                self.tree = {
                                'split_attribute': split_attribute,
                                'split_value': split_value
                            }
                self.tree['left'] = DecisionTree((self.max_depth - 1))
                self.tree['right'] = DecisionTree((self.max_depth - 1))
                self.tree['left'].learn(X_left, y_left)
                self.tree['right'].learn(X_right, y_right)

    def classify(self, record):
                
        if (not isinstance(self.tree, dict)):
            return self.tree
        else:
            if (isinstance(record[self.tree['split_attribute']], str)):
                if record[self.tree['split_attribute']] == self.tree['split_value']:
                    return self.tree['left'].classify(record)
                else:
                    return self.tree['right'].classify(record)
            else:
                if record[self.tree['split_attribute']] <= self.tree['split_value']:
                    return self.tree['left'].classify(record)
                else:
                    return self.tree['right'].classify(record)
