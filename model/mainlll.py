from scipy.io import loadmat
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from model import *
import warnings
import pandas as pd
from analysislll import *


def data_loader():
    print("Loading MNIST Dataset...")
    mnist_path = "./mnist-original.mat"
    mnist_raw = loadmat(mnist_path)
    mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }
    X, y = mnist["data"], mnist["target"]
    print("Load MNIST Dataset Successfully!")


    print("Preparing Training set and Testing set...")
    # X_train, X_test, y_train, y_test = X[:60000],X[60000:],y[:60000],y[60000:]

    Ｘ_test, y_test = X[60000:], y[60000:]

    X_train = X[1000:3000]
    X_train = np.vstack((X_train,X[7000:9000]))
    X_train = np.vstack((X_train,X[13000:15000]))
    X_train = np.vstack((X_train,X[19000:21000]))
    X_train = np.vstack((X_train,X[25000:27000]))
    X_train = np.vstack((X_train,X[31000:33000]))
    X_train = np.vstack((X_train,X[37000:39000]))
    X_train = np.vstack((X_train,X[43000:45000]))
    X_train = np.vstack((X_train,X[49000:51000]))
    X_train = np.vstack((X_train,X[55000:57000]))
    

    y_train = y[1000:3000]
    y_train = np.hstack((y_train,y[7000:9000]))
    y_train = np.hstack((y_train,y[13000:15000]))
    y_train = np.hstack((y_train,y[19000:21000]))
    y_train = np.hstack((y_train,y[25000:27000]))
    y_train = np.hstack((y_train,y[31000:33000]))
    y_train = np.hstack((y_train,y[37000:39000]))
    y_train = np.hstack((y_train,y[43000:45000]))
    y_train = np.hstack((y_train,y[49000:51000]))
    y_train = np.hstack((y_train,y[55000:57000]))

    shuffle_index = np.random.permutation(20000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
    print("Training set and Testing set are ready!")

    return X_train, X_test, y_train, y_test


def plot_roc_curve(fpr, tpr, label):
    plt.plot(fpr, tpr, linewidth = 2, label = label)
    #plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 0.4, 0.6, 1])
    plt.legend(loc = 'best')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


def plot_pr_curve(ps, rs, label):
    plt.plot(rs, ps, linewidth = 2, label = label)
    #plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0.6, 1, 0.6, 1])
    plt.legend(loc = 'best')
    plt.xlabel('Recalls')
    plt.ylabel('Precisions')


if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    
    res = []

    clfer = [
        "k-Nearest Neighbor", 
        "Linear Regression",
        "Logistic Regression",
        "Support Vector Machine",
        "Random Forest",
        "Adaptive Boosting",
        "Gradient Boosting",
        "Multi-Layer Perceptron"
    ]

    print("We have", clfer)

    for cl in range(0, 10):
        X_train, X_test, y_train, y_test = data_loader()

        train = zip(y_train, X_train)
        test = zip(y_test, X_test)
        print("Now task", cl, "...")
        
        rcl = []
        # format: [
        # precision,       [0]
        # recall,          [1]
        # f1-score,        [2]
        # fpr,             [3]
        # tpr,             [4]
        # roc_auc,         [5]
        # precisions,      [6]
        # recalls,         [7]
        # pr_auc           [8]
        # running_time     [9] 
        # ]

        if (cl == 0):
            train_cl = [x for x in train if (x[0]==0 or x[0]==1)]
            test_cl = [x for x in test if (x[0]==0 or x[0]==1)]
            y_train_c = [x[0] for x in train_cl]
            X_train_cl = [x[1] for x in train_cl]
            y_test_c = [x[0] for x in test_cl]
            X_test_cl = [x[1] for x in test_cl]
            y_train_cl = []
            for i in y_train_c:
                if i == cl: y_train_cl.append(True)
                else: y_train_cl.append(False)
            y_test_cl = []
            for i in y_train_c:
                if i == cl: y_test_cl.append(True)
                else: y_test_cl.append(False)
            
        
        elif (cl == 1):  
            train_cl = [x for x in train if (x[0]==0 or x[0]==1)]
            test_cl = [x for x in test if (x[0]==0 or x[0]==1)]
            y_train_c = [x[0] for x in train_cl]
            X_train_cl = [x[1] for x in train_cl]
            y_test_c = [x[0] for x in test_cl]
            X_test_cl = [x[1] for x in test_cl]
            y_train_cl = []
            for i in y_train_c:
                if i == cl: y_train_cl.append(True)
                else: y_train_cl.append(False)
            y_test_cl = []
            for i in y_train_c:
                if i == cl: y_test_cl.append(True)
                else: y_test_cl.append(False)

            

        elif (cl == 2):
            train_cl = [x for x in train if (x[0]==0 or x[0]==1 or x[0]==2)]
            test_cl = [x for x in test if (x[0]==0 or x[0]==1 or x[0]==2)]
            y_train_c = [x[0] for x in train_cl]
            X_train_cl = [x[1] for x in train_cl]
            y_test_c = [x[0] for x in test_cl]
            X_test_cl = [x[1] for x in test_cl]
            y_train_cl = []
            for i in y_train_c:
                if i == cl: y_train_cl.append(True)
                else: y_train_cl.append(False)
            y_test_cl = []
            for i in y_train_c:
                if i == cl: y_test_cl.append(True)
                else: y_test_cl.append(False)

        elif (cl == 3):
            train_cl = [x for x in train if (x[0]==0 or x[0]==1 or x[0]==2 or x[0]==3)]
            test_cl = [x for x in test if (x[0]==0 or x[0]==1 or x[0]==2 or x[0]==3)]
            y_train_c = [x[0] for x in train_cl]
            X_train_cl = [x[1] for x in train_cl]
            y_test_c = [x[0] for x in test_cl]
            X_test_cl = [x[1] for x in test_cl]
            y_train_cl = []
            for i in y_train_c:
                if i == cl: y_train_cl.append(True)
                else: y_train_cl.append(False)
            y_test_cl = []
            for i in y_train_c:
                if i == cl: y_test_cl.append(True)
                else: y_test_cl.append(False)

        elif (cl == 4):
            train_cl = [x for x in train if (x[0]==0 or x[0]==1 or x[0]==2 or x[0]==3 or x[0]==4)]
            test_cl = [x for x in test if (x[0]==0 or x[0]==1 or x[0]==2 or x[0]==3 or x[0]==4)]
            y_train_c = [x[0] for x in train_cl]
            X_train_cl = [x[1] for x in train_cl]
            y_test_c = [x[0] for x in test_cl]
            X_test_cl = [x[1] for x in test_cl]
            y_train_cl = []
            for i in y_train_c:
                if i == cl: y_train_cl.append(True)
                else: y_train_cl.append(False)
            y_test_cl = []
            for i in y_train_c:
                if i == cl: y_test_cl.append(True)
                else: y_test_cl.append(False)

        elif (cl == 5):
            train_cl = [x for x in train if (x[0]==0 or x[0]==1 or x[0]==2 or x[0]==3 or x[0]==4 or x[0]==5)]
            test_cl = [x for x in test if (x[0]==0 or x[0]==1 or x[0]==2 or x[0]==3 or x[0]==4 or x[0]==5)]
            y_train_c = [x[0] for x in train_cl]
            X_train_cl = [x[1] for x in train_cl]
            y_test_c = [x[0] for x in test_cl]
            X_test_cl = [x[1] for x in test_cl]
            y_train_cl = []
            for i in y_train_c:
                if i == cl: y_train_cl.append(True)
                else: y_train_cl.append(False)
            y_test_cl = []
            for i in y_train_c:
                if i == cl: y_test_cl.append(True)
                else: y_test_cl.append(False)

        elif (cl == 6):
            train_cl = [x for x in train if (x[0]==0 or x[0]==1 or x[0]==2 or x[0]==3 or x[0]==4 or x[0]==5 or x[0]==6)]
            test_cl = [x for x in test if (x[0]==0 or x[0]==1 or x[0]==2 or x[0]==3 or x[0]==4 or x[0]==5 or x[0]==6)]
            y_train_c = [x[0] for x in train_cl]
            X_train_cl = [x[1] for x in train_cl]
            y_test_c = [x[0] for x in test_cl]
            X_test_cl = [x[1] for x in test_cl]
            y_train_cl = []
            for i in y_train_c:
                if i == cl: y_train_cl.append(True)
                else: y_train_cl.append(False)
            y_test_cl = []
            for i in y_train_c:
                if i == cl: y_test_cl.append(True)
                else: y_test_cl.append(False)

        elif (cl == 7):
            train_cl = [x for x in train if (x[0]==0 or x[0]==1 or x[0]==2 or x[0]==3 or x[0]==4 or x[0]==5 or x[0]==6 or x[0]==7)]
            test_cl = [x for x in test if (x[0]==0 or x[0]==1 or x[0]==2 or x[0]==3 or x[0]==4 or x[0]==5 or x[0]==6 or x[0]==7)]
            y_train_c = [x[0] for x in train_cl]
            X_train_cl = [x[1] for x in train_cl]
            y_test_c = [x[0] for x in test_cl]
            X_test_cl = [x[1] for x in test_cl]
            y_train_cl = []
            for i in y_train_c:
                if i == cl: y_train_cl.append(True)
                else: y_train_cl.append(False)
            y_test_cl = []
            for i in y_train_c:
                if i == cl: y_test_cl.append(True)
                else: y_test_cl.append(False)

        elif (cl == 8):
            train_cl = [x for x in train if (x[0]==0 or x[0]==1 or x[0]==2 or x[0]==3 or x[0]==4 or x[0]==5 or x[0]==6 or x[0]==7 or x[0]==8)]
            test_cl = [x for x in test if (x[0]==0 or x[0]==1 or x[0]==2 or x[0]==3 or x[0]==4 or x[0]==5 or x[0]==6 or x[0]==7 or x[0]==8)]
            y_train_c = [x[0] for x in train_cl]
            X_train_cl = [x[1] for x in train_cl]
            y_test_c = [x[0] for x in test_cl]
            X_test_cl = [x[1] for x in test_cl]
            y_train_cl = []
            for i in y_train_c:
                if i == cl: y_train_cl.append(True)
                else: y_train_cl.append(False)
            y_test_cl = []
            for i in y_train_c:
                if i == cl: y_test_cl.append(True)
                else: y_test_cl.append(False)

        elif (cl == 9):
            train_cl = [x for x in train if (x[0]==0 or x[0]==1 or x[0]==2 or x[0]==3 or x[0]==4 or x[0]==5 or x[0]==6 or x[0]==7 or x[0]==8 or x[0]==9)]
            test_cl = [x for x in test if (x[0]==0 or x[0]==1 or x[0]==2 or x[0]==3 or x[0]==4 or x[0]==5 or x[0]==6 or x[0]==7 or x[0]==8 or x[0]==9)]
            y_train_c = [x[0] for x in train_cl]
            X_train_cl = [x[1] for x in train_cl]
            y_test_c = [x[0] for x in test_cl]
            X_test_cl = [x[1] for x in test_cl]
            y_train_cl = []
            for i in y_train_c:
                if i == cl: y_train_cl.append(True)
                else: y_train_cl.append(False)
            y_test_cl = []
            for i in y_train_c:
                if i == cl: y_test_cl.append(True)
                else: y_test_cl.append(False)

        

        r = kNN(X_train_cl, X_test_cl, y_train_cl, y_test_cl)
        rcl.append(r)

        r = SGD(X_train_cl, X_test_cl, y_train_cl, y_test_cl)
        rcl.append(r)

        r = LOG(X_train_cl, X_test_cl, y_train_cl, y_test_cl)
        rcl.append(r)

        r = SVM(X_train_cl, X_test_cl, y_train_cl, y_test_cl)
        rcl.append(r)

        r = RaF(X_train_cl, X_test_cl, y_train_cl, y_test_cl)
        rcl.append(r)

        r = AdaBST(X_train_cl, X_test_cl, y_train_cl, y_test_cl)
        rcl.append(r)

        r = GDBT(X_train_cl, X_test_cl, y_train_cl, y_test_cl)
        rcl.append(r)

        r = MLP(X_train_cl, X_test_cl, y_train_cl, y_test_cl)
        rcl.append(r)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in clfer:
            j = clfer.index(i)
            plot_roc_curve(rcl[j][3], rcl[j][4], i)

        ax.set_aspect('equal')
        title = "ROC Comparison for Task_" + str(cl)
        plt.title(title)
        figname = "resultlll/per_task/ROC_Task_" + str(cl)
        plt.savefig(figname)
        print("Visualization for Classifiers on Task", cl, "saved!")
        #plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in clfer:
            j = clfer.index(i)
            plot_pr_curve(rcl[j][6], rcl[j][7], i)

        ax.set_aspect('equal')
        title = "PR Comparison for Task_" + str(cl)
        plt.title(title)
        figname = "resultlll/per_task/PR_Task_" + str(cl)
        plt.savefig(figname)
        print("Visualization for Classifiers on Task", cl, "saved!")
        #plt.show()

        res.append(rcl)

    
    resa = np.array(res)
    np.save('resultlll/report/binaryclass.npy', resa)
    print("Results saved as np array!")

    result = pd.DataFrame(
        columns = clfer, 
        index = range(0, 10), 
        data = res
    )
    result.to_csv('resultlll/report/binaryclass.csv')
    print("Result saved as .csv file!")

    analysislll(res, clfer)
    






