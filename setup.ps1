# Setup script: create virtualenv and install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Setup complete. Activate with: .\.venv\Scripts\Activate.ps1"
