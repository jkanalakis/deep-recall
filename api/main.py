#!/usr/bin/env python3
# api/main.py

import json
import os
import time
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from pydantic import BaseModel, Field

from api.auth.routes import router as auth_router
# Import custom modules
from api.endpoints import inference, memory
from api.middleware.auth import authenticate_request
from api.middleware.logging import LoggingMiddleware
from api.middleware.prometheus import PrometheusMiddleware

# Create FastAPI app
app = FastAPI(
    title="deep-recall API Gateway",
    description="A scalable hyper personalized memory framework for LLMs",
    version="0.1.0",
)

# Setup security
security = HTTPBearer()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(PrometheusMiddleware)

# Include routers for each service
app.include_router(
    memory.router,
    prefix="/api/memory",
    tags=["memory"],
    dependencies=[Security(security)],
)

app.include_router(
    inference.router,
    prefix="/api/inference",
    tags=["inference"],
    dependencies=[Security(security)],
)

# Include authentication router
app.include_router(auth_router, prefix="/api/auth", tags=["auth"])


# Health check endpoint
@app.get("/health", include_in_schema=False)
async def health_check():
    return {"status": "ok", "timestamp": time.time()}


# Root endpoint
@app.get("/")
async def root():
    return {
        "name": "deep-recall API Gateway",
        "version": "0.1.0",
        "documentation": "/docs",
        "health": "/health",
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
