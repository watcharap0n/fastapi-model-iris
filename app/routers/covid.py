from fastapi import APIRouter
import pandas as pd
import requests

router = APIRouter()


@router.get('/initialized')
async def initialized_covid():
    path = 'https://covid19.ddc.moph.go.th/api/Cases/round-1to2-by-provinces'
    res = requests.get(path)
    data = res.json()
    df = pd.DataFrame(data)
    df.to_csv('app/static/dataset_covid.csv')
    return data


@router.get('/', summary='Read covid', description='api read data csv on pandas')
async def read_covid():
    path = 'app/static/dataset_covid.csv'
    df = pd.read_csv(path)
    data = df.to_dict('records')
    return data


@router.post('/')
async def add_covid():
    pass
