## Run NetraVision Locally

### CMD
```cmd
cd /d D:\NetraVision
.venv\Scripts\activate.bat
python -m pip install -r backend\requirements.txt
python backend\app.py
```

### PowerShell
```powershell
cd D:\NetraVision
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
python backend\app.py
```

### Open in Browser
- Home: http://localhost:5000/
- Prediction: http://localhost:5000/predict.html

Keep the terminal open while using the website. Press `Ctrl + C` to stop the server.
