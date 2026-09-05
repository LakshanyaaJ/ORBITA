"""ORBITA Backend FastAPI Application Server.

Usage:
    uvicorn backend.server:app --host 0.0.0.0 --port 8000
    or via main.py:
    python main.py --mode web
"""
from orbita.backend.api import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)
