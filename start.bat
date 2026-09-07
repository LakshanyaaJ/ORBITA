@echo off
setlocal enabledelayedexpansion

title ORBITA Launcher

echo =====================================================================
echo           ORBITA - Human Activity Recognition Copilot
echo =====================================================================
echo.

:: Resolve workspace root
set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

:: Locate Python executable (.venv, venv, or system python)
if exist "%ROOT_DIR%.venv\Scripts\python.exe" (
    set "PYTHON_CMD=%ROOT_DIR%.venv\Scripts\python.exe"
    echo [*] Using virtual environment: .venv
) else if exist "%ROOT_DIR%venv\Scripts\python.exe" (
    set "PYTHON_CMD=%ROOT_DIR%venv\Scripts\python.exe"
    echo [*] Using virtual environment: venv
) else (
    set "PYTHON_CMD=python"
    echo [*] Using system Python
)

:: Verify Node.js / npm is available
where npm >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Node.js and npm are required but were not found in PATH.
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

:: Verify Python is available
%PYTHON_CMD% --version >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is required but was not found in PATH or virtualenv.
    echo Please install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

:: Free ports if any lingering zombie processes exist
echo [*] Freeing ports 8000, 8443, and 5173/5174...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8000,8443,5173,5174 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }" 2>nul

echo [*] Starting ORBITA Backend (FastAPI on Port 8000)...
start "ORBITA - Backend" cmd /k "title ORBITA - Backend && cd /d "%ROOT_DIR%" && "%PYTHON_CMD%" main.py --mode web --scenario A"

:: Wait dynamically for backend to finish loading models and open port 8000
echo [*] Waiting for Backend models to load and port 8000 to be ready...
powershell -NoProfile -Command "for ($i=0; $i -lt 30; $i++) { try { $client = New-Object System.Net.Sockets.TcpClient('127.0.0.1', 8000); if ($client.Connected) { $client.Close(); exit 0 } } catch {}; Start-Sleep -Seconds 1 }; exit 0"

echo [*] Starting ORBITA Frontend (Vite on Port 5173)...
start "ORBITA - Frontend" cmd /k "title ORBITA - Frontend && cd /d "%ROOT_DIR%frontend" && npm run dev"

:: Wait dynamically for Vite to open port 5173
echo [*] Waiting for Frontend server on port 5173...
powershell -NoProfile -Command "for ($i=0; $i -lt 15; $i++) { try { $client = New-Object System.Net.Sockets.TcpClient('127.0.0.1', 5173); if ($client.Connected) { $client.Close(); exit 0 } } catch {}; Start-Sleep -Seconds 1 }; exit 0"

echo.
echo =====================================================================
echo                     ORBITA Services Launched!
echo =====================================================================
echo.
echo   [1] React Dashboard:      http://localhost:5173
echo   [2] Backend API:          http://localhost:8000
echo   [3] Live Video Stream:    http://localhost:8000/video_feed
echo   [4] Phone Webcam Web App: http://localhost:8000/cam
echo.
echo   Individual service consoles are running in separate titled windows.
echo   To terminate services, close their windows or run stop.bat.
echo =====================================================================
echo.
echo Opening Dashboard in your browser...
start http://localhost:5173

echo.
echo Launcher finished. Press any key to close this launcher window...
exit
