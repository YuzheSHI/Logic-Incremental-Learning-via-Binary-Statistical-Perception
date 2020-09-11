import time
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier



def kNN(X_train, X_test, y_train_cl, y_test_cl):
    start = time.time()
    print("Now evaluate the k-Nearest Neighbor Classifier...")

    knn_clf = KNeighborsClassifier()
    
    y_train_pred = cross_val_predict(
        knn_clf,
        X_train,
        y_train_cl,
        cv = 3
    )
    y_proba = cross_val_predict(
        knn_clf, 
        X_train, 
        y_train_cl, 
        cv = 3,
        method = "predict_proba"
    )
    y_scores = y_proba[:,1]
    p = precision_score(y_train_cl, y_train_pred)
    r = recall_score(y_train_cl, y_train_pred)
    f = f1_score(y_train_cl, y_train_pred)
    fpr, tpr, thresholds = roc_curve(y_train_cl, y_scores)
    roc = roc_auc_score(y_train_cl, y_scores)
    ps, rs, ths = precision_recall_curve(y_train_cl, y_scores)
    pr = auc(rs, ps)

    end = time.time()
    t = end - start

    print("Precision = ", p)
    print("Recall = ", r)
    print("F1-score = ", f)
    print("Finished in ", t, "s")

    return [p, r, f, fpr, tpr, roc, ps, rs, pr, t]


def SGD(X_train, X_test, y_train_cl, y_test_cl):
    start = time.time()
    print("Now evaluate the Linear Regression Classifier...")
    
    sgd_clf = SGDClassifier(random_state = 42)
    
    y_train_pred = cross_val_predict(
        sgd_clf,
        X_train,
        y_train_cl,
        cv = 3
    )
    y_scores = cross_val_predict(
        sgd_clf, 
        X_train, 
        y_train_cl, 
        cv = 3, 
        method = "decision_function"
    )
    p = precision_score(y_train_cl, y_train_pred)
    r = recall_score(y_train_cl, y_train_pred)
    f = f1_score(y_train_cl, y_train_pred)
    fpr, tpr, thresholds = roc_curve(y_train_cl, y_scores)
    roc = roc_auc_score(y_train_cl, y_scores)
    ps, rs, ths = precision_recall_curve(y_train_cl, y_scores)
    pr = auc(rs, ps)

    end = time.time()
    t = end - start

    print("Precision = ", p)
    print("Recall = ", r)
    print("F1-score = ", f)
    print("Finished in ", t, "s")

    return [p, r, f, fpr, tpr, roc, ps, rs, pr, t]


def LOG(X_train, X_test, y_train_cl, y_test_cl):
    start = time.time()
    print("Now evaluate the Logistic Regression Classifier...")
    
    log_clf = LogisticRegression()
    
    y_train_pred = cross_val_predict(
        log_clf,
        X_train,
        y_train_cl,
        cv = 3
    )
    y_scores = cross_val_predict(
        log_clf, 
        X_train, 
        y_train_cl, 
        cv = 3, 
        method = "decision_function"
    )
    p = precision_score(y_train_cl, y_train_pred)
    r = recall_score(y_train_cl, y_train_pred)
    f = f1_score(y_train_cl, y_train_pred)
    fpr, tpr, thresholds = roc_curve(y_train_cl, y_scores)
    roc = roc_auc_score(y_train_cl, y_scores)
    ps, rs, ths = precision_recall_curve(y_train_cl, y_scores)
    pr = auc(rs, ps)

    end = time.time()
    t = end - start

    print("Precision = ", p)
    print("Recall = ", r)
    print("F1-score = ", f)
    print("Finished in ", t, "s")

    return [p, r, f, fpr, tpr, roc, ps, rs, pr, t]


def SVM(X_train, X_test, y_train_cl, y_test_cl):
    start = time.time()
    print("Now evaluate the Support Vector Machine Classifier...")
    
    svm_clf = svm.SVC(random_state = 42)
    
    y_train_pred = cross_val_predict(
        svm_clf,
        X_train,
        y_train_cl,
        cv = 3
    )
    y_scores = cross_val_predict(
        svm_clf, 
        X_train, 
        y_train_cl, 
        cv = 3, 
        method = "decision_function"
    )
    p = precision_score(y_train_cl, y_train_pred)
    r = recall_score(y_train_cl, y_train_pred)
    f = f1_score(y_train_cl, y_train_pred)
    fpr, tpr, thresholds = roc_curve(y_train_cl, y_scores)
    roc = roc_auc_score(y_train_cl, y_scores)
    ps, rs, ths = precision_recall_curve(y_train_cl, y_scores)
    pr = auc(rs, ps)

    end = time.time()
    t = end - start

    print("Precision = ", p)
    print("Recall = ", r)
    print("F1-score = ", f)
    print("Finished in ", t, "s")

    return [p, r, f, fpr, tpr, roc, ps, rs, pr, t]


def RaF(X_train, X_test, y_train_cl, y_test_cl):
    start = time.time()
    print("Now evaluate the Random Forest Classifier...")

    forest_clf = RandomForestClassifier(random_state = 42)
    
    y_train_pred = cross_val_predict(
        forest_clf,
        X_train,
        y_train_cl,
        cv = 3
    )
    y_proba_scores = cross_val_predict(
        forest_clf,
        X_train, 
        y_train_cl, 
        cv = 3,
        method = "predict_proba"
    )
    y_scores = y_proba_scores[:,1]

    p = precision_score(y_train_cl, y_train_pred)
    r = recall_score(y_train_cl, y_train_pred)
    f = f1_score(y_train_cl, y_train_pred)
    fpr, tpr, thresholds = roc_curve(y_train_cl, y_scores)
    roc = roc_auc_score(y_train_cl, y_scores)
    ps, rs, ths = precision_recall_curve(y_train_cl, y_scores)
    pr = auc(rs, ps)

    end = time.time()
    t = end - start

    print("Precision = ", p)
    print("Recall = ", r)
    print("F1-score = ", f)
    print("Finished in ", t, "s")

    return [p, r, f, fpr, tpr, roc, ps, rs, pr, t]


def AdaBST(X_train, X_test, y_train_cl, y_test_cl):
    start = time.time()
    print("Now evaluate the Adaptive Boosting Classifier...")
    
    bst_clf = AdaBoostClassifier(random_state = 42)
    
    y_train_pred = cross_val_predict(
        bst_clf,
        X_train,
        y_train_cl,
        cv = 3
    )
    y_scores = cross_val_predict(
        bst_clf, 
        X_train, 
        y_train_cl, 
        cv = 3, 
        method = "decision_function"
    )
    p = precision_score(y_train_cl, y_train_pred)
    r = recall_score(y_train_cl, y_train_pred)
    f = f1_score(y_train_cl, y_train_pred)
    fpr, tpr, thresholds = roc_curve(y_train_cl, y_scores)
    roc = roc_auc_score(y_train_cl, y_scores)
    ps, rs, ths = precision_recall_curve(y_train_cl, y_scores)
    pr = auc(rs, ps)

    end = time.time()
    t = end - start

    print("Precision = ", p)
    print("Recall = ", r)
    print("F1-score = ", f)
    print("Finished in ", t, "s")

    return [p, r, f, fpr, tpr, roc, ps, rs, pr, t]


def GDBT(X_train, X_test, y_train_cl, y_test_cl):
    start = time.time()
    print("Now evaluate the Gradient Boosting Classifier...")
    
    gdbt_clf = GradientBoostingClassifier(random_state = 42)
    
    y_train_pred = cross_val_predict(
        gdbt_clf,
        X_train,
        y_train_cl,
        cv = 3
    )
    y_scores = cross_val_predict(
        gdbt_clf, 
        X_train, 
        y_train_cl, 
        cv = 3, 
        method = "decision_function"
    )
    p = precision_score(y_train_cl, y_train_pred)
    r = recall_score(y_train_cl, y_train_pred)
    f = f1_score(y_train_cl, y_train_pred)
    fpr, tpr, thresholds = roc_curve(y_train_cl, y_scores)
    roc = roc_auc_score(y_train_cl, y_scores)
    ps, rs, ths = precision_recall_curve(y_train_cl, y_scores)
    pr = auc(rs, ps)

    end = time.time()
    t = end - start

    print("Precision = ", p)
    print("Recall = ", r)
    print("F1-score = ", f)
    print("Finished in ", t, "s")

    return [p, r, f, fpr, tpr, roc, ps, rs, pr, t]


def MLP(X_train, X_test, y_train_cl, y_test_cl):
    start = time.time()
    print("Now evaluate the Multi-Layer Perceptron...")
    
    mlp_clf = MLPClassifier(random_state = 42)
    
    y_train_pred = cross_val_predict(
        mlp_clf,
        X_train,
        y_train_cl,
        cv = 3
    )
    y_proba_scores = cross_val_predict(
        mlp_clf,
        X_train, 
        y_train_cl, 
        cv = 3,
        method = "predict_proba"
    )
    y_scores = y_proba_scores[:,1]
    p = precision_score(y_train_cl, y_train_pred)
    r = recall_score(y_train_cl, y_train_pred)
    f = f1_score(y_train_cl, y_train_pred)
    fpr, tpr, thresholds = roc_curve(y_train_cl, y_scores)
    roc = roc_auc_score(y_train_cl, y_scores)
    ps, rs, ths = precision_recall_curve(y_train_cl, y_scores)
    pr = auc(rs, ps)

    end = time.time()
    t = end - start

    print("Precision = ", p)
    print("Recall = ", r)
    print("F1-score = ", f)
    print("Finished in ", t, "s")

    return [p, r, f, fpr, tpr, roc, ps, rs, pr, t]



