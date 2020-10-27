# basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np


# sklearn imports
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# our code
import utils

from knn import KNN

from naive_bayes import NaiveBayes

from decision_stump import DecisionStumpErrorRate, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
# from random_forest import RandomForest

from kmeans import Kmeans
from sklearn.cluster import DBSCAN

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question


    if question == "1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_val, y_val = dataset["Xtest"], dataset["ytest"]
        model = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_val)
        te_error = np.mean(y_pred != y_val)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

    elif question == "1.1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_val, y_val = dataset["Xtest"], dataset["ytest"]

        depths = np.arange(1,16)
        train_errors = np.zeros(depths.size)
        val_errors = np.zeros(depths.size)

        for i, depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=depth, criterion='entropy', random_state=1)
            model.fit(X, y)

            y_pred_train = model.predict(X)
            train_errors[i] = np.mean(y_pred_train != y)

            y_pred_val = model.predict(X_val)
            val_errors[i] = np.mean(y_pred_val != y_val)

        plt.plot(depths, train_errors, label="Training Error")
        plt.plot(depths, val_errors, label="Testing Error")

        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q1_1_tree_errors.pdf")
        plt.savefig(fname)

    elif question == '1.2':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        n, d = X.shape

        X_train, X_val = np.vsplit(X, 2)
        y_train, y_val = np.split(y, 2)

        depths = np.arange(1, 16)
        train_errors = np.zeros(depths.size)
        val_errors = np.zeros(depths.size)

        for i, depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=depth, criterion='entropy', random_state=1)
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            train_errors[i] = np.mean(y_pred_train != y_train)

            y_pred_val = model.predict(X_val)
            val_errors[i] = np.mean(y_pred_val != y_val)

        plt.plot(depths, train_errors, label="Training Error")
        plt.plot(depths, val_errors, label="Validation Error")

        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q1_2_tree_errors.pdf")
        plt.savefig(fname)



    elif question == '2.2':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]
        groupnames = dataset["groupnames"]
        wordlist = dataset["wordlist"]




    elif question == '2.3':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]

        print("d = %d" % X.shape[1])
        print("n = %d" % X.shape[0])
        print("t = %d" % X_valid.shape[0])
        print("Num classes = %d" % len(np.unique(y)))

        model = NaiveBayes(num_classes=4)
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes (ours) validation error: %.3f" % v_error)

    

    elif question == '3':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']

        ks = [1, 3, 10]
        training_errors = np.zeros(len(ks))
        testing_errors = np.zeros(len(ks))

        for i, k in enumerate(ks):
            model = KNN(k)
            model.fit(X, y)

            y_pred_train = model.predict(X)
            training_errors[i] = np.mean(y_pred_train != y)

            y_pred_test = model.predict(Xtest)
            testing_errors[i] = np.mean(y_pred_test != ytest)

        print("Training:", training_errors)
        print("Testing:", testing_errors)

        own_model = KNN(k=1)
        own_model.fit(X, y)

        sci_model = KNeighborsClassifier(n_neighbors=1)
        sci_model.fit(X, y)

        utils.plotClassifier(own_model, Xtest, ytest)
        fname = os.path.join("..", "figs", "q3_ownKNN_boundary.pdf")
        plt.savefig(fname)

        utils.plotClassifier(sci_model, Xtest, ytest)
        fname = os.path.join("..", "figs", "q3_scikitKNN_boundary.pdf")
        plt.savefig(fname)


    elif question == '4':
        dataset = load_dataset('vowel.pkl')
        X = dataset['X']
        y = dataset['y']
        X_val = dataset['Xtest']
        y_val = dataset['ytest']
        print("\nn = %d, d = %d\n" % X.shape)

        def evaluate_model(model):
            model.fit(X,y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_val)
            te_error = np.mean(y_pred != y_val)
            print("    Training error: %.3f" % tr_error)
            print("    Testing error: %.3f" % te_error)

        print("Decision tree info gain")
        evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))



    elif question == '5':
        X = load_dataset('clusterData.pkl')['X']

        model = Kmeans(k=4)
        model.fit(X)
        y = model.predict(X)
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet")

        fname = os.path.join("..", "figs", "kmeans_basic.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == '5.1':
        X = load_dataset('clusterData.pkl')['X']

        errors = np.zeros(50)
        models = []
        for i in range(50):
            model = Kmeans(k=4)
            model.fit(X)

            models.append(model)
            errors[i] = model.error(X)

        best_model = models[np.argmin(errors)]

        y = best_model.predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")
        print("Best model error rate: {}".format(best_model.error(X)))


        fname = os.path.join("..", "figs", "q5_1_best_model.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)



    elif question == '5.2':
        X = load_dataset('clusterData.pkl')['X']

        ks = np.arange(1, 11)
        best_errors = np.zeros(len(ks))

        for k in ks:
            errors = np.zeros(50)
            models = []
            for i in range(50):
                model = Kmeans(k=k)
                model.fit(X)

                models.append(model)
                errors[i] = model.error(X)

            best_model = models[np.argmin(errors)]
            best_errors[k - 1] = best_model.error(X)

        plt.plot(ks, best_errors, label="Error rate")
        fname = os.path.join("..", "figs", "q5_2_error_k.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)



    elif question == '5.3':
        X = load_dataset('clusterData2.pkl')['X']

        model = DBSCAN(eps=16, min_samples=3)
        y = model.fit_predict(X)

        print("Labels (-1 is unassigned):", np.unique(model.labels_))
        
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet", s=5)
        fname = os.path.join("..", "figs", "density.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        
        plt.xlim(-25,25)
        plt.ylim(-15,30)
        fname = os.path.join("..", "figs", "density2.png")
        plt.savefig(fname)
        print("Figure saved as '%s'" % fname)
        
    else:
        print("Unknown question: %s" % question)
