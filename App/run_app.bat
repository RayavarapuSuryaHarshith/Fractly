@echo off
echo ========================================
echo  BONE FRACTURE DETECTION - HYBRID AI
echo  93.2%% Accuracy Medical Grade System
echo ========================================
echo.
echo Starting Streamlit application...
echo.
cd /d "%~dp0"
python -m streamlit run app_hybrid.py --server.port 8501 --server.address localhost
echo.
echo Application stopped.
pause