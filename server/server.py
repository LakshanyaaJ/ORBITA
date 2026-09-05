"""ORBITA Backend FastAPI Application Server.

Usage:
    uvicorn server.server:app --host 0.0.0.0 --port 8000
    or via main.py:
    python main.py --mode web
"""
from core_ai.backend.api import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.server:app", host="0.0.0.0", port=8000, reload=True)
