import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import pandas as pd
from main import *

def analysislll(res, clfer):
    print("Now analysis the results...")
    # res[0, 1, 2, 3, 4, 5, 6, 7, 8, 9][][]
    # res[][# "k-Nearest Neighbor", 
        # "Linear Regression",
        # "Logistic Regression",
        # "Support Vector Machine",
        # "Random Forest",
        # "Adaptive Boosting",
        # "Gradient Boosting",
        # "Multi-Layer Perceptron"][]
    #  res[][] [
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

    
    for i in clfer:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        j = clfer.index(i)
        for n in range(0, 10):
            plot_roc_curve(res[n][j][3], res[n][j][4], "Class_" + str(n))

        ax.set_aspect('equal')
        title = "ROC Comparison for Classifier " + i
        plt.title(title)
        figname = "resultlll/per_classifier/ROC_Classifier_" + i
        plt.savefig(figname)
        print("Visualization for Classes on Classifier", i, "saved!")


    for i in clfer:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        j = clfer.index(i)
        for n in range(0, 10):
            plot_pr_curve(res[n][j][6], res[n][j][7], "Class_" + str(n))

        ax.set_aspect('equal')
        title = "PR Comparison for Classifier " + i
        plt.title(title)
        figname = "resultlll/per_classifier/PR_Classifier_" + i
        plt.savefig(figname)
        print("Visualization for Classes on Classifier", i, "saved!")


    m = []
    v = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in clfer:
        k = clfer.index(i)
        y = []
        for j in range(0, 10):
            roc = res[j][k][5]
            y.append(roc)
        
        plt.plot([0,1,2,3,4,5,6,7,8,9], y, marker = 'o', label = i, markersize = 6)
        m.append(np.mean(y))
        v.append(np.var(y))

    plt.xticks([0,1,2,3,4,5,6,7,8,9])
    plt.legend(loc = 'best')
    plt.title("AUC of ROC Comparison for Binary Classifiers")
    plt.xlabel("Classes")
    plt.ylabel("AUC of ROC")
    figname = "resultlll/per_classifier/AUC_ROC"
    plt.savefig(figname)
    print("Visualization for AUC_ROC on Classifiers saved!")

    m = zip(m, clfer)
    smean = sorted(m, reverse = True)
    v = zip(v, clfer)
    svar = sorted(v)
    np.save('resultlll/report/AUC_ROC_mean.npy', np.array(smean))
    np.save('resultlll/report/AUC_ROC_var.npy', np.array(svar))
    print("Results saved as np array!")
    pd.DataFrame(data = smean).to_csv('resultlll/report/AUC_ROC_mean.csv')
    pd.DataFrame(data = svar).to_csv('resultlll/report/AUC_ROC_var.csv')
    print("Results saved as .csv file!")


    m = []
    v = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in clfer:
        k = clfer.index(i)
        y = []
        for j in range(0, 10):
            pr = res[j][k][8]
            y.append(pr)
        
        plt.plot([0,1,2,3,4,5,6,7,8,9], y, marker = 'o', label = i, markersize = 6)
        m.append(np.mean(y))
        v.append(np.var(y))

    plt.xticks([0,1,2,3,4,5,6,7,8,9])
    plt.legend(loc = 'best')
    plt.title("AUC of PR Comparison for Binary Classifiers")
    plt.xlabel("Classes")
    plt.ylabel("AUC of PR")
    figname = "resultlll/per_classifier/AUC_PR"
    plt.savefig(figname)
    print("Visualization for AUC_PR on Classifiers saved!")

    m = zip(m, clfer)
    smean = sorted(m, reverse = True)
    v = zip(v, clfer)
    svar = sorted(v)
    np.save('resultlll/report/AUC_PR_mean.npy', np.array(smean))
    np.save('resultlll/report/AUC_PR_var.npy', np.array(svar))
    print("Results saved as np array!")
    pd.DataFrame(data = smean).to_csv('resultlll/report/AUC_PR_mean.csv')
    pd.DataFrame(data = svar).to_csv('resultlll/report/AUC_PR_var.csv')
    print("Results saved as .csv file!")


    m = []
    v = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in clfer:
        k = clfer.index(i)
        y = []
        for j in range(0, 10):
            p = res[j][k][0]
            y.append(p)
        
        plt.plot([0,1,2,3,4,5,6,7,8,9], y, marker = 'o', label = i, markersize = 6)
        m.append(np.mean(y))
        v.append(np.var(y))

    plt.xticks([0,1,2,3,4,5,6,7,8,9])
    plt.legend(loc = 'best')
    plt.title("Precision Comparison for Binary Classifiers")
    plt.xlabel("Classes")
    plt.ylabel("Precision")
    figname = "resultlll/per_classifier/Precision"
    plt.savefig(figname)
    print("Visualization for Precision on Classifiers saved!")

    m = zip(m, clfer)
    smean = sorted(m, reverse = True)
    v = zip(v, clfer)
    svar = sorted(v)
    np.save('resultlll/report/P_mean.npy', np.array(smean))
    np.save('resultlll/report/P_var.npy', np.array(svar))
    print("Results saved as np array!")
    pd.DataFrame(data = smean).to_csv('resultlll/report/P_mean.csv')
    pd.DataFrame(data = svar).to_csv('resultlll/report/P_var.csv')
    print("Results saved as .csv file!")


    m = []
    v = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in clfer:
        k = clfer.index(i)
        y = []
        for j in range(0, 10):
            r = res[j][k][1]
            y.append(r)
        
        plt.plot([0,1,2,3,4,5,6,7,8,9], y, marker = 'o', label = i, markersize = 6)
        m.append(np.mean(y))
        v.append(np.var(y))

    plt.xticks([0,1,2,3,4,5,6,7,8,9])
    plt.legend(loc = 'best')
    plt.title("Recall Comparison for Binary Classifiers")
    plt.xlabel("Classes")
    plt.ylabel("Recall")
    figname = "resultlll/per_classifier/Recall"
    plt.savefig(figname)
    print("Visualization for Recall on Classifiers saved!")

    m = zip(m, clfer)
    smean = sorted(m, reverse = True)
    v = zip(v, clfer)
    svar = sorted(v)
    np.save('resultlll/report/R_mean.npy', np.array(smean))
    np.save('resultlll/report/R_var.npy', np.array(svar))
    print("Results saved as np array!")
    pd.DataFrame(data = smean).to_csv('resultlll/report/R_mean.csv')
    pd.DataFrame(data = svar).to_csv('resultlll/report/R_var.csv')
    print("Results saved as .csv file!")


    m = []
    v = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in clfer:
        k = clfer.index(i)
        y = []
        for j in range(0, 10):
            r = res[j][k][2]
            y.append(r)
        
        plt.plot([0,1,2,3,4,5,6,7,8,9], y, marker = 'o', label = i, markersize = 6)
        m.append(np.mean(y))
        v.append(np.var(y))

    plt.xticks([0,1,2,3,4,5,6,7,8,9])
    plt.legend(loc = 'best')
    plt.title("F1-score Comparison for Binary Classifiers")
    plt.xlabel("Classes")
    plt.ylabel("F1-score")
    figname = "resultlll/per_classifier/F1-score"
    plt.savefig(figname)
    print("Visualization for F1-score on Classifiers saved!")

    m = zip(m, clfer)
    smean = sorted(m, reverse = True)
    v = zip(v, clfer)
    svar = sorted(v)
    np.save('resultlll/report/F1_mean.npy', np.array(smean))
    np.save('resultlll/report/F1_var.npy', np.array(svar))
    print("Results saved as np array!")
    pd.DataFrame(data = smean).to_csv('resultlll/report/F1_mean.csv')
    pd.DataFrame(data = svar).to_csv('resultlll/report/F1_var.csv')
    print("Results saved as .csv file!")


    t = []
    for i in clfer:
        k = clfer.index(i)
        y = []
        for j in range(0, 10):
            r = res[j][k][9]
            y.append(r)
        
        t.append(np.mean(y))

    tz = zip(t, clfer)
    smean = sorted(tz)
    np.save('resultlll/report/time.npy', np.array(smean))
    print("Results saved as np array!")
    pd.DataFrame(data = smean).to_csv('resultlll/report/time.csv')
    print("Results saved as .csv file!")


    return