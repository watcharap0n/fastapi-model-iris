from fastapi import FastAPI
from app.routers import users, covid

app = FastAPI()

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
