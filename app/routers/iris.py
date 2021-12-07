from fastapi import APIRouter
from typing import Optional, Any, List
from sklearn import datasets
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pydantic import BaseModel, conlist

router = APIRouter()


class Iris(BaseModel):
    data: List[conlist(float, min_items=2, max_items=2)]


async def plot_features_iris(feature: int, preprocessing: Optional[bool] = False):
    data = datasets.load_iris()
    x = range(50)
    X = StandardScaler().fit_transform(data.data[:, :])
    p1, p2, p3 = data.data[:50, feature], data.data[50:100, feature], data.data[100:, feature]
    plt.scatter(x, X[:50, feature] if preprocessing else p1, color='red')
    plt.scatter(x, X[50:100, feature] if preprocessing else p2, color='blue')
    plt.scatter(x, X[100:, feature] if preprocessing else p3, color='green')
    plt.savefig('app/static/graph/{}.jpg'.format('petal_width' if feature == 3 else 'petal_length'))
    return plt


async def dynamic_predict(model, preprocessing: Optional[bool] = False, X_one: Optional[Any] = None) -> tuple:
    data = datasets.load_iris()
    X = StandardScaler().fit_transform(data.data[:, 2:]) if preprocessing else data.data[:, 2:]
    X_train, X_test, y_train, y_test = train_test_split(X, data.target, test_size=0.2, stratify=data.target)
    X_test = X_test if X_one is None else X_one
    clf = model
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    log_proba = clf.predict_proba(X_test)
    return clf, predict, y_test if X_one is None else None, log_proba


@router.get('/')
async def read_iris(
        desc: Optional[str] = None,
):
    data = datasets.load_iris()
    if desc:
        if desc == 'feature_names':
            res = {'status': True, 'description': 'feature names', 'data': data.feature_names}
            return res
        elif desc == 'target_names':
            res = {'status': True, 'description': 'feature target names', 'data': data.target_names.tolist()}
            return res
        elif desc == 'target':
            res = {'status': True, 'description': 'feature target value', 'data': data.target.tolist()}
            return res
    else:
        res = {'status': True, 'description': 'feature names', 'data': data.data.tolist()}
        return res


@router.get('/{plot}')
async def _class(plot: Optional[str] = None, pp: Optional[bool] = False):
    if plot == 'petal_length':
        _plt = await plot_features_iris(3, preprocessing=(True if pp else False))
        _plt.close()
        return {'status': True, 'message': 'plot graph petal_width', 'data': '/static/graph/petal_length.jpg'}
    elif plot == 'petal_width':
        _plt = await plot_features_iris(2, preprocessing=(True if pp else False))
        _plt.close()
        return {'status': True, 'message': 'plot graph petal_width', 'data': '/static/graph/petal_width.jpg'}


@router.post('/prediction/{model}')
async def prediction(payload: Iris, model: Optional[str] = None, all: Optional[bool] = False,
                     pp: Optional[bool] = False):
    __class = ['setosa', 'versicolor', 'virginica']
    if model == 'knn':
        if all:
            _, predict, y_test, log_proba = await dynamic_predict(model=KNeighborsClassifier(),
                                                                  preprocessing=(True if pp else False))
            acc = accuracy_score(y_test, predict)
            clf_report = classification_report(y_test, predict)
            lst = [__class[x] for x in predict]
            res = {'status': True, 'message': 'predict model KNN', 'predict': predict.tolist(),
                   'predict_name': lst, 'log_proba': log_proba.tolist(),
                   'y_true': y_test.tolist(), 'acc': float(acc),
                   'clf_report': clf_report}
            return res
        elif all is False:
            payload = payload.dict()
            _, predict, y_test, log_proba = await dynamic_predict(model=KNeighborsClassifier(), X_one=payload['data'],
                                                                  preprocessing=(True if pp else False))
            lst = [__class[x] for x in predict]
            res = {'status': True, 'message': 'predict model KNN', 'predict': predict.tolist(),
                   'predict_name': lst, 'log_proba': log_proba.tolist(), 'y_true': None,
                   'acc': None,
                   'clf_report': None}
            return res
