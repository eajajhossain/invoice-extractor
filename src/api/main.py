from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(title="Invoice Intelligence API")
app.include_router(router)
