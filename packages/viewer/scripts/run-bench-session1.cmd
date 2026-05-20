@echo off
REM SPDX-License-Identifier: Apache-2.0
REM Bench runner invoked by Windows Task Scheduler in the interactive
REM session (session 1) so Chrome's WebGPU/Dawn can reach the real D3D12
REM GPU. SSH lands in session 0, where dxcore returns no adapters.
setlocal
set HOME=%USERPROFILE%
set PATH=%APPDATA%\npm;%PATH%
cd /d %USERPROFILE%\Catetus
node packages\viewer\scripts\run-bench-windows.mjs > %USERPROFILE%\bench-stdout.log 2> %USERPROFILE%\bench-stderr.log
echo %ERRORLEVEL% > %USERPROFILE%\bench-exit.log
endlocal
