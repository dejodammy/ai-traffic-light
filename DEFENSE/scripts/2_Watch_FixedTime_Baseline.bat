@echo off
REM ============================================================================
REM  DEFENSE DEMO 2 - The "dumb" FIXED-TIME baseline on the SAME junction
REM  Real Ikeja junction, same traffic, but the lights just cycle on a timer
REM  with no awareness of the queues. Run this AFTER demo 1 to compare. The
REM  average queue/wait at the end will be higher than the AI's.
REM ============================================================================
title AI Traffic Light - FIXED-TIME BASELINE (real Ikeja 4-phase junction)

if "%SUMO_HOME%"=="" set "SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo"

cd /d "%~dp0\..\..\src"

echo.
echo  Launching the FIXED-TIME (dumb timer) controller on the same junction...
echo  Watch how the queues build up because the timer ignores them.
echo.

"..\venv\Scripts\python.exe" watch_model.py ^
    --scenario lagos_4ph_random ^
    --fixed ^
    --delay 300

echo.
echo  Compare the "Average total queue" here against Demo 1 (the AI).
pause
