@echo off
rem Usage:  install.bat nvidia   (RTX 3070 Ti etc.)
rem         install.bat amd      (RX 9070 XT etc. - whisper.cpp Vulkan)
rem         install.bat amd-rocm (RX 9070 XT - whisper.cpp ROCm build, ~500 MB)
rem         install.bat cpu
setlocal
cd /d "%~dp0"
set TARGET=%1
if "%TARGET%"=="" set TARGET=nvidia

where py >nul 2>nul && (set PY=py -3) || (set PY=python)
if not exist .venv (
    echo Creating virtual environment...
    %PY% -m venv .venv || goto :fail
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul

if /i "%TARGET%"=="nvidia" (
    python -m pip install -e ".[nvidia]" || goto :fail
) else (
    python -m pip install -e . || goto :fail
)
if /i "%TARGET%"=="amd" python scripts\setup_whisper_cpp.py --variant vulkan --model large-v3 || goto :fail
if /i "%TARGET%"=="amd-rocm" python scripts\setup_whisper_cpp.py --variant rocm-gfx120X --model large-v3 || goto :fail

echo.
python -m lecture_notes --doctor
echo.
echo Done. Set your Claude API key once:   setx ANTHROPIC_API_KEY "sk-ant-..."
echo Then double-click start.bat
exit /b 0

:fail
echo Install failed.
exit /b 1
