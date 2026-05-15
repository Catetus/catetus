@echo off
REM Real-scene + synthetic bench, schtask-invoked variant.
setlocal
set HOME=%USERPROFILE%
set PATH=%APPDATA%\npm;%PATH%
set SF_BENCH_PLY_DIR=%USERPROFILE%\SplatForge\.bench-scenes
cd /d %USERPROFILE%\SplatForge
node packages\viewer\scripts\run-bench-windows.mjs > %USERPROFILE%\bench-stdout-real.log 2> %USERPROFILE%\bench-stderr-real.log
echo %ERRORLEVEL% > %USERPROFILE%\bench-exit-real.log
endlocal
