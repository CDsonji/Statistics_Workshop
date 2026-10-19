# 🛠️ Workshop Setup Commands & Virtual Environment Guide

Follow these commands to create an isolated Python virtual environment and install all necessary dependencies for the traffic data analysis workshop.

---

## 1. Environment Setup Commands

### Option A: Windows (Command Prompt / PowerShell)

```cmd
# 1. Ensure venv is available (standard Python installations on Windows include venv by default)
python -m ensurepip --default-pip

# 2. Create virtual environment named '.venv'
python -m venv .venv

# 3. Activate the virtual environment
# For Command Prompt:
.venv\Scripts\activate.bat
# For PowerShell:
.venv\Scripts\Activate.ps1

# 4. Upgrade pip and install libraries
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Register the virtual environment kernel with Jupyter
python -m ipykernel install --user --name=stats_workshop --display-name "Python (Stats Workshop)"
```

### Option B: Linux (Ubuntu / Debian / Fedora)

```bash
# 1. Install system prerequisites (python3-venv and python3-pip)
# For Ubuntu/Debian:
sudo apt update && sudo apt install -y python3-venv python3-pip

# For Fedora/RHEL:
# sudo dnf install -y python3-pip

# 2. Create virtual environment named '.venv'
python3 -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Upgrade pip and install libraries
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Register the virtual environment kernel with Jupyter
python -m ipykernel install --user --name=stats_workshop --display-name "Python (Stats Workshop)"
```

### Option C: macOS (Intel & Apple Silicon) 
```bash
# 1. Ensure pip and standard environment modules are available
python3 -m ensurepip --upgrade

# 2. Create virtual environment named '.venv'
python3 -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Upgrade pip and install libraries
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Register the virtual environment kernel with Jupyter
python -m ipykernel install --user --name=stats_workshop --display-name "Python (Stats Workshop)"
```