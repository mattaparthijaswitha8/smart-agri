@echo off
echo Installing requirements...
pip install -r requirements.txt
echo Starting server at http://127.0.0.1:5000
start http://127.0.0.1:5000
python app.py
pause
