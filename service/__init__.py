"""
Service package – exposes api.app for uvicorn / gunicorn.
"""

from .api import app

__all__ = ["app"]
