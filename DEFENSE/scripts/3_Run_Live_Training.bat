@echo off
REM ============================================================================
REM  DEFENSE DEMO 3 - Show TRAINING actually happening in SUMO
REM  Real Ikeja junction. Runs a SHORT live training session (15 episodes) with
REM  the SUMO window open so the panel can see the agent learning by trial and
REM  error. It warm-starts from the imitation pre-trained model.
REM  NOTE: this only demonstrates the *process*. The real model behind the
REM  results was trained for 200 episodes and is already saved in
REM  results\lagos_4ph_random_rl\. This demo writes to a SEPARATE folder
REM  (results\defense_live_train\) and does NOT overwrite it.
REM ============================================================================
title AI Traffic Light - LIVE TRAINING demo (real Ikeja 4-phase junction)

if "%SUMO_HOME%"=="" set "SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo"

cd /d "%~dp0\..\..\src"

echo.
echo  Starting a short LIVE training run (15 episodes) with the SUMO GUI open.
echo  You will see the agent explore (random-looking moves at first) and improve.
echo  Close the SUMO window or press Ctrl+C to stop early at any time.
echo.

"..\venv\Scripts\python.exe" main.py train ^
    --scenario lagos_4ph_random ^
    --episodes 15 ^
    --pretrained-checkpoint "..\results\lagos_4ph_random_pretrain\expert_pretrained_dqn_model.pt" ^
    --epsilon-start 0.2 ^
    --output-dir "..\results\defense_live_train" ^
    --gui

echo.
echo  Live training demo complete. The full 200-episode model is the one in
echo  results\lagos_4ph_random_rl\ that the published results come from.
pause
