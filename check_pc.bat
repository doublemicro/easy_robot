@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PY=python
where python >nul 2>&1 || set PY=py
(echo 사용한 파이썬 명령: %PY% & where %PY%) > check_pc_error.txt 2>&1
%PY% -m tools.check_pc 2>> check_pc_error.txt
if errorlevel 1 (echo. & echo [에러] check_pc_error.txt 내용: & type check_pc_error.txt)
echo.
echo 결과 파일: %~dp0check_pc_result.txt
pause
