from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # fit a model
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import GradientBoostingClassifier


def random_forest_model(feats,labs):
    "fits a random forest model on the data given"

    X_train, X_test, y_train, y_test=train_test_split(feats,labs,test_size=0.33, random_state=42,shuffle=True)
    clf=RandomForestRegressor(50,n_jobs=-1)
    clf.fit(X_train,y_train)
    preds=clf.predict(X_test)
    print("model score: ", clf.score(X_test,y_test))

    print("mean absolute error: ", mean_absolute_error(y_test,preds))
    return clf



def gradient_boosting_model(feats,labs):
    "fits a gradient boosting model on the data given"

    X_train, X_test, y_train, y_test=train_test_split(feats,labs,test_size=0.33, random_state=42,shuffle=True)
    clf = GradientBoostingClassifier(n_estimators=60)
    clf.fit(X_train,y_train)
    preds=clf.predict(X_test)
    print("model score: ", clf.score(X_test,y_test))

    print("mean absolute error: ", mean_absolute_error(y_test,preds))
    return clf