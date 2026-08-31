@echo off
echo Starting FastAPI Backend on http://127.0.0.1:8000 ...
call .venv\Scripts\activate.bat
uvicorn main:app --reload --port 8000
pause
