@echo off
setlocal enabledelayedexpansion

title ORBITA Terminator

echo =====================================================================
echo                Stopping ORBITA Services...
echo =====================================================================
echo.

powershell -NoProfile -Command "Get-Process -Id (Get-NetTCPConnection -LocalPort 8000,5173,5174 -State Listen -ErrorAction SilentlyContinue).OwningProcess -ErrorAction SilentlyContinue | Stop-Process -Force"

echo.
echo =====================================================================
echo                    ORBITA Services Stopped.
echo =====================================================================
timeout /t 2 /nobreak >nul
exit /b 0
