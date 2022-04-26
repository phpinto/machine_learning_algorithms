from decision_tree import DecisionTree
import csv
import numpy as np
import ast
from datetime import datetime

class RandomForest(object):

    num_trees = 0
    decision_trees = []

    bootstraps_datasets = []
    bootstraps_labels = []

    def __init__(self, num_trees):

        self.num_trees = num_trees
        self.decision_trees = [DecisionTree(max_depth=10) for i in range(num_trees)]
        
    def _bootstrapping(self, XX, n):

        samples = [] # sampled dataset
        labels = []  # class labels for the sampled records
        
        for _ in range(n):
            idx = np.random.randint(n)
            samples.append(XX[idx][:-1])
            labels.append(XX[idx][-1])
        
        return (samples, labels)

    def bootstrapping(self, XX):
        # Initializing the bootstap datasets for each tree
        for _ in range(self.num_trees):
            data_sample, data_label = self._bootstrapping(XX, len(XX))
            self.bootstraps_datasets.append(data_sample)
            self.bootstraps_labels.append(data_label)

    def fitting(self):

        for i in range(self.num_trees):
            self.decision_trees[i].learn(self.bootstraps_datasets[i], self.bootstraps_labels[i])

    def voting(self, X):
        y = []

        for record in X:
       
            votes = []
            bag_votes = []

            for i in range(len(self.bootstraps_datasets)):
                dataset = self.bootstraps_datasets[i]
                
                if record not in dataset:
                    OOB_tree = self.decision_trees[i]
                    effective_vote = OOB_tree.classify(record)
                    
                    votes.append(effective_vote)
                else:
                    BAG_tree = self.decision_trees[i]
                    effective_vote = BAG_tree.classify(record)
                    bag_votes.append(effective_vote)


            counts = np.bincount(votes)
            bag_counts = np.bincount(bag_votes)

            if len(counts) == 0:
                y = np.append(y, np.argmax(bag_counts))
            else:
                y = np.append(y, np.argmax(counts))
                
        return y

    def user(self):
        return 'ppinto3'

def main():

    # start time 
    start = datetime.now()

    X = list()
    y = list()
    XX = list()  # Contains data features and data labels
    numerical_cols = set([i for i in range(0, 10)])  # indices of numeric attributes (columns)

    # Loading data set
    print("reading the data")
    with open("Churn.csv") as f:
        next(f, None)
        for line in csv.reader(f, delimiter=","):
            xline = []
            for i in range(len(line)):
                if i in numerical_cols:
                    xline.append(ast.literal_eval(line[i]))
                else:
                    xline.append(line[i])

            X.append(xline[:-1])
            y.append(xline[-1])
            XX.append(xline[:])

    # TODO: Initialize according to your implementation
    # VERY IMPORTANT: Minimum forest_size should be 10
    forest_size = 50

    # Initializing a random forest.
    randomForest = RandomForest(forest_size)

    # printing the name
    print("__Name: " + randomForest.user()+"__")

    # Creating the bootstrapping datasets
    print("creating the bootstrap datasets")
    randomForest.bootstrapping(XX)

    # Building trees in the forest
    print("fitting the forest")
    randomForest.fitting()

    # Calculating an unbiased error estimation of the random forest
    # based on out-of-bag (OOB) error estimate.
    y_predicted = randomForest.voting(X)

    # Comparing predicted and true labels
    results = [prediction == truth for prediction, truth in zip(y_predicted, y)]

    # Accuracy
    accuracy = float(results.count(True)) / float(len(results))

    print("accuracy: %.4f" % accuracy)
    print("OOB estimate: %.4f" % (1 - accuracy))

    # end time
    print("Execution time: " + str(datetime.now() - start))


if __name__ == "__main__":
    main()