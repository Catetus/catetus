@echo off
REM EWA-dilation sweep bench (novel-2-renderer). Skips the synthetic and
REM base real-scene benches because the dilation sweep does its own
REM measurements per dilation value.
setlocal
set HOME=%USERPROFILE%
set PATH=%APPDATA%\npm;%PATH%
set SF_BENCH_PLY_DIR=%USERPROFILE%\SplatForge\.bench-scenes
set SF_SKIP_SYNTH=1
set SF_SKIP_REAL_BASE=1
set SF_DILATION_SWEEP=1
cd /d %USERPROFILE%\SplatForge
node packages\viewer\scripts\run-bench-windows.mjs > %USERPROFILE%\bench-stdout-dilation.log 2> %USERPROFILE%\bench-stderr-dilation.log
echo %ERRORLEVEL% > %USERPROFILE%\bench-exit-dilation.log
endlocal
