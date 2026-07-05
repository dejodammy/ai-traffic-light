@echo off
REM ============================================================================
REM  YOLO DEMO (LIVE) - prove the vehicle DETECTION + COUNTING works
REM  Opens your webcam, draws a green box on every vehicle YOLOv8 detects, shows
REM  a live VEHICLES count on the video, AND prints the count in this terminal.
REM
REM  WHAT TO POINT IT AT:
REM   - out a window at the road / car park, OR
REM   - a phone or second screen playing any traffic video (e.g. a YouTube clip)
REM  Either way the panel sees real vehicles being detected and counted live.
REM
REM  Press 'q' in the video window to quit.
REM  (If your webcam isn't index 0, edit --source below to 1 or 2.)
REM ============================================================================
title YOLO Vehicle Counting - LIVE WEBCAM

cd /d "%~dp0\..\..\src"

echo.
echo  Starting live YOLO vehicle counting on the webcam...
echo  Point the camera at vehicles (or a screen playing traffic) and watch the
echo  count update here and on the video window. Press 'q' to stop.
echo.

"..\venv\Scripts\python.exe" yolo_count_demo.py --source 0 --confidence 0.35

pause
