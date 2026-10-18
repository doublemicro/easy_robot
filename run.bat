@echo off
chcp 65001 >nul
cd /d "%~dp0"
python play.py %* 2> run_error.txt
if errorlevel 1 (type run_error.txt & pause)
