from fastapi import FastAPI

from app.api.routes.router import api_router

toy_app = FastAPI(title="Titanic API", version="0.0.1")
toy_app.include_router(api_router, prefix="/api")
