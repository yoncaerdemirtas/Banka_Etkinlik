import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score,\
    roc_auc_score, auc, f1_score, recall_score, RocCurveDisplay, roc_curve
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("banka_classification.csv")
X = df.drop(["target"], axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)


def decision_tree():
    dt_model = DecisionTreeClassifier()
    #GridSearch for Cross Validation
    params = {
        'max_depth': [2, 3, 5, 10],
        'min_samples_leaf': [5, 10, 20, 50],
        'criterion': ["gini", "entropy"]
    }
    grid_search = GridSearchCV(estimator=dt_model,
                               param_grid=params,
                               cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")
    grid_search.fit(X_train, y_train)
    score_df = pd.DataFrame(grid_search.cv_results_)
    #print(score_df)
    print(grid_search.best_estimator_)

    dt_model_tuned = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5).fit(X_train, y_train)
    y_pred = dt_model_tuned.predict(X_test)
    y_score = dt_model_tuned.predict_proba(X_test)[:, 1]
    model_performance(y_test, y_pred, y_score, "Karar agaci")
    fig = plt.figure(figsize=(10, 10))
    plot_tree(dt_model_tuned, class_names =["0", "1"], filled=True, proportion=True, feature_names=X.columns)
    fig.savefig("siniflandirma_agaci.png")
    plot_feature_importance(dt_model_tuned.feature_importances_,X.columns,'Karar Agaci')
    return y_score


def xg_boost():
    xg = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xg_model = xg.fit(X_train, y_train)
    y_score = xg_model.predict_proba(X_test)[:, 1]
    y_pred = xg_model.predict(X_test)
    model_performance(y_test, y_pred, y_score, "Xg_Boost")
    plot_feature_importance(xg_model.feature_importances_,X.columns,'XGBoost')
    return y_score


def random_forest():
    rf = RandomForestClassifier().fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_score = rf.predict_proba(X_test)[:, 1]
    model_performance(y_test, y_pred, y_score, "Rassal Ormanlar")
    plot_feature_importance(rf.feature_importances_,X.columns,'Rassal Ormanlar')
    return y_score


def plot_feature_importance(importance,names,model_type):
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    print(fi_df)
    a = plt.figure(figsize=(6,7))
    sns.barplot(x=fi_df['feature_names'], y=fi_df['feature_importance'])
    #Add chart labels
    plt.title(model_type)
    plt.xlabel('Değişken Adı',)
    plt.ylabel('Önem Düzeyi')
    plt.xticks(rotation=90)
    plt.savefig(str(model_type))
    plt.show()
    plt.close(a)


def model_performance(y_t, y_p, y_s, model_name):
    print(model_name + ": karışıklık matrisi: " + str(confusion_matrix(y_t, y_p)))
    print(model_name + ": dogruluk: " + str(accuracy_score(y_t, y_p)))
    print(model_name + ": kesinlik: " + str(precision_score(y_t, y_p)))
    print(model_name + ": Duyarlılık: " + str(recall_score(y_t, y_p)))
    print(model_name + ": f1: " + str(f1_score(y_t, y_p)))
    print(model_name + ": auc: " + str(roc_auc_score(y_t, y_s)))
    fpr, tpr, thresholds = roc_curve(y_t, y_s)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=model_name)
    display.plot()
    plt.show()


if __name__ == "__main__":
    y_score1 = decision_tree()
    y_score2 = random_forest()
    y_score3 = xg_boost()

    #print roc curves in one graphic
    fpr1, tpr1, thresholds1 = roc_curve(y_test, y_score1)
    fpr2, tpr2, thresholds2 = roc_curve(y_test, y_score2)
    fpr3, tpr3, thresholds3 = roc_curve(y_test, y_score3)
    auc1 = roc_auc_score(y_test, y_score1)
    auc2 = roc_auc_score(y_test, y_score2)
    auc3 = roc_auc_score(y_test, y_score3)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr1, tpr1, label="Karar Ağacı: AUC = " +str("%.2f" % auc1))
    plt.plot(fpr2, tpr2, label="Rassal Ormanlar: AUC = " +str("%.2f" % auc2))
    plt.plot(fpr3, tpr3, label="XGBoost: AUC = "+ str("%.2f" % auc3))

    plt.legend()
    plt.xlabel("Yanlış Pozitif Oranı")
    plt.ylabel("Doğru Pozitif Oranı")
    plt.title('ROC Eğrisi')
    plt.show()
