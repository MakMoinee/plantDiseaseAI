@echo off
REM Activate virtual environment
call .\Scripts\activate

REM Ask user for camera IP
set /p CAMERA_IP=Enter ESP32 camera IP (e.g., [http://192.168.1.5:4747](http://192.168.1.5:4747)):

REM Run Python app with the provided IP
python api3.py --ip %CAMERA_IP%

pause
