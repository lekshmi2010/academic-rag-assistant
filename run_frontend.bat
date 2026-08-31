@echo off
echo Starting Streamlit Frontend on http://localhost:8501 ...
call .venv\Scripts\activate.bat
streamlit run app.py
pause
