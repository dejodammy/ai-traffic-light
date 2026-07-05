@echo off
REM ============================================================================
REM  DEFENSE DEMO 1 - Watch the TRAINED AI control the lights live in SUMO
REM  Runs on the REAL Ikeja junction (imported from OpenStreetMap).
REM  You will SEE the AI pick green phases and clear the queues. The queue
REM  length is printed each step.
REM ============================================================================
title AI Traffic Light - TRAINED MODEL (real Ikeja 4-phase junction, live SUMO)

REM -- Make sure SUMO can be found (edit this line if SUMO is installed elsewhere)
if "%SUMO_HOME%"=="" set "SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo"

REM -- Jump to the project's src folder no matter where this .bat is run from
cd /d "%~dp0\..\..\src"

echo.
echo  Launching the TRAINED DQN agent on the REAL Ikeja junction...
echo  (4-phase protected signal: each approach gets its own green.)
echo  A SUMO window will open. Press the green PLAY arrow if it does not auto-run.
echo.

"..\venv\Scripts\python.exe" watch_model.py ^
    --scenario lagos_4ph_random ^
    --checkpoint "..\results\lagos_4ph_random_rl\best_dqn_model.pt" ^
    --delay 300

echo.
echo  Demo finished. The "Average total queue" printed above is the AI's score.
pause
