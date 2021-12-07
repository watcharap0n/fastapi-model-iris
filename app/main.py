from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routers import users, covid, iris

app = FastAPI()
app.mount('/static', StaticFiles(directory='app/static'), name='static')

app.include_router(
    users.router,
    prefix='/users',
    tags=['users'],
    responses={418: {'description': "I'm teapot!"}}
)

app.include_router(
    covid.router,
    prefix='/covid',
    tags=['covid'],
    responses={418: {'description': "I'm teapot!"}}
)

app.include_router(
    iris.router,
    prefix='/iris',
    tags=['iris'],
    responses={418: {'description': "I'm teapot"}}
)

