@echo off
title SpineSync AI - PostureGuard Pro Analytics Suite
echo ============================================================
echo   Starting SpineSync AI - PostureGuard Pro HAR Dashboard...
echo ============================================================
cd /d "%~dp0"
python -m streamlit run app.py
pause
