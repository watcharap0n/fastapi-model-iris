"""
Machine Leaning Dataset Iris
- features
    - sepal length in cm
    - sepal width in cm
    - petal length in cm
    - petal width in cm
- class:
    - Iris-Setosa
    - Iris-Versicolour
    - Iris-Virginica

    ============== ==== ==== ======= ===== ====================
                    Min  Max   Mean    SD   Class Correlation
    ============== ==== ==== ======= ===== ====================
    sepal length:   4.3  7.9   5.84   0.83    0.7826
    sepal width:    2.0  4.4   3.05   0.43   -0.4194
    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
    ============== ==== ==== ======= ===== ====================
"""

from typing import Optional, Any
from fastapi import APIRouter
from sklearn import datasets
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from model import iris

router = APIRouter()


async def plot_features_iris(feature: int, preprocessing: Optional[bool] = False):
    """

    :param feature:
    :param preprocessing:
    :return:
        -> func
    """
    data = datasets.load_iris()
    _x = range(50)
    scaler = StandardScaler()
    x_scaler = scaler.fit_transform(data.data[:, :])
    class_a, class_b, class_c = data.data[:50, feature], \
                                data.data[50:100, feature], \
                                data.data[100:, feature]
    plt.scatter(_x, x_scaler[:50, feature] if preprocessing else class_a, color='red')
    plt.scatter(_x, x_scaler[50:100, feature] if preprocessing else class_b, color='blue')
    plt.scatter(_x, x_scaler[100:, feature] if preprocessing else class_c, color='green')
    petal_condition = 'petal_width' if feature == 3 else 'petal_length'
    plt.savefig(f'static/graph/{petal_condition}.jpg')
    return plt


async def dynamic_predict(
        model,
        preprocessing: Optional[bool] = False,
        x_one: Optional[Any] = None
) -> tuple:
    """

    :param model:
    :param preprocessing:
    :param x_one:
    :return:
        -> tuple
    """
    data = datasets.load_iris()
    scaler = StandardScaler()
    x_scaler = scaler.fit_transform(data.data[:, 2:]) \
        if preprocessing else data.data[:, 2:]
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaler, data.target,
        test_size=0.2,
        stratify=data.target
    )
    x_test = x_test if x_one is None else x_one
    clf = model
    clf.fit(x_train, y_train)
    predict = clf.predict(x_test)
    log_proba = clf.predict_proba(x_test)
    return clf, predict, y_test if x_one is None else None, log_proba


@router.get('/')
async def read_iris(
        desc: Optional[str] = None,
):
    """

    :param desc:
    :return:
        -> json
    """
    print(desc)
    data = datasets.load_iris()
    if desc:
        print(desc)
        if desc == 'feature_names':
            res = {'status': True, 'description': 'feature names',
                   'data': data.feature_names}
            return res

        if desc == 'target_names':
            res = {'status': True, 'description': 'feature target names',
                   'data': data.target_names.tolist()}
            return res

        if desc == 'target':
            res = {'status': True, 'description': 'feature target value',
                   'data': data.target.tolist()}
            return res

        res = {'status': False, 'description': 'feature not invalid name!', 'data': None}
        return res

    res = {'status': True, 'description': 'all feature', 'data': data.data.tolist()}
    return res


@router.get('/{plot}')
async def _class(plot: Optional[str] = None, preprocess: Optional[bool] = False):
    """

    :param plot:
    :param preprocess:
    :return:
        -> json
    """
    if plot == 'petal_length':
        _plt = await plot_features_iris(3, preprocessing=preprocess)
        _plt.close()
        return {'status': True, 'message': 'plot graph petal_length',
                'data': '/static/graph/petal_length.jpg'}

    if plot == 'petal_width':
        _plt = await plot_features_iris(2, preprocessing=preprocess)
        _plt.close()
        return {'status': True, 'message': 'plot graph petal_width',
                'data': '/static/graph/petal_width.jpg'}

    return {'status': False, 'message': 'plot graph not invalid name!',
            'data': None}


@router.post('/prediction/{model}')
async def prediction(
        payload: iris.Iris,
        model: Optional[str] = None,
        all_select: Optional[bool] = False,
        preprocess: Optional[bool] = False
):
    """

    :param preprocess:
    :param payload:
    :param model:
    :param all_select:
    :return:
        -> json
    """
    __class = ['setosa', 'versicolor', 'virginica']
    if model == 'knn':
        if all_select:
            _, predict, y_test, log_proba = await dynamic_predict(
                model=KNeighborsClassifier(),
                preprocessing=preprocess
            )
            acc = accuracy_score(y_test, predict)
            clf_report = classification_report(y_test, predict)
            lst = [__class[x] for x in predict]
            res = {'status': True, 'message': 'predict model KNN', 'predict': predict.tolist(),
                   'predict_name': lst, 'log_proba': log_proba.tolist(),
                   'y_true': y_test.tolist(), 'acc': float(acc),
                   'clf_report': clf_report}
            return res

        if all_select is False:
            payload = payload.dict()
            _, predict, y_test, log_proba = await dynamic_predict(
                model=KNeighborsClassifier(),
                x_one=payload['data'],
                preprocessing=preprocess)
            lst = [__class[x] for x in predict]
            res = {'status': True, 'message': 'predict model KNN', 'predict': predict.tolist(),
                   'predict_name': lst, 'log_proba': log_proba.tolist(), 'y_true': None,
                   'acc': None,
                   'clf_report': None}
            return res

    return {'status': False, 'message': 'model not invalid name!'}
