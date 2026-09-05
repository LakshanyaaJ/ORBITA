@echo off
setlocal enabledelayedexpansion

title ORBITA Terminator

echo =====================================================================
echo                Stopping ORBITA Services...
echo =====================================================================
echo.

powershell -NoProfile -Command "$ports = @(8000, 8443, 5173, 5174); $pids = (Get-NetTCPConnection -LocalPort $ports -State Listen -ErrorAction SilentlyContinue).OwningProcess; if ($pids) { Get-Process -Id $pids -ErrorAction SilentlyContinue | Stop-Process -Force }" 2>nul

echo.
echo =====================================================================
echo                    ORBITA Services Stopped.
echo =====================================================================
ping 127.0.0.1 -n 2 >nul
exit /b 0
