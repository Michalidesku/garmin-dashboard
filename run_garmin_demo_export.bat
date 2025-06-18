@echo off
cd /d "C:\Users\skriv\Desktop\Garmin-demo"
set GARMIN_EMAIL=skrivanekm1@gmail.com
set GARMIN_PASSWORD=Halloween23
echo Running Garmin export script...>> "C:\Users\skriv\Desktop\Garmin-demo\run_log.txt"
"C:\Users\skriv\Desktop\Garmin-demo\.venv\Scripts\python.exe" "C:\Users\skriv\Desktop\Garmin-demo\garmindemo.py">> "C:\Users\skriv\Desktop\Garmin-demo\run_log.txt" 2>&1
echo Done. >> "C:\Users\skriv\Desktop\Garmin-demo\run_log.txt"
