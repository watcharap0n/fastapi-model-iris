from fastapi import APIRouter, HTTPException
from typing import Optional
from sklearn import datasets

router = APIRouter()


@router.get('/')
async def read_iris(
        desc: Optional[str] = None
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


@router.post('/')
async def prediction():
    pass
