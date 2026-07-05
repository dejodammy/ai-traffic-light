@echo off
REM ============================================================================
REM  YOLO DEMO (CANNED) - guaranteed to work with NO webcam needed
REM  Runs YOLOv8 on a real traffic video that ships with this pack. It draws a
REM  box on each detected vehicle, shows the live VEHICLES count on the video,
REM  and prints the count in this terminal as cars pass through.
REM
REM  Use this as a reliable fallback if the webcam demo can't be run.
REM
REM  TIP: you can also drag-and-drop ANY traffic photo or video file onto this
REM  .bat to run YOLO on that file instead.
REM
REM  Press 'q' in the video window to quit.
REM ============================================================================
title YOLO Vehicle Counting - SAMPLE TRAFFIC VIDEO

cd /d "%~dp0\..\..\src"

set "INPUT=%~1"
if "%INPUT%"=="" set "INPUT=..\sumo_scenarios1\sample_traffic_video.mp4"

echo.
echo  Running YOLO vehicle counting on:  %INPUT%
echo  Watch the count change as vehicles pass. Press 'q' to stop.
echo.

"..\venv\Scripts\python.exe" yolo_count_demo.py --source "%INPUT%" --confidence 0.3

pause
