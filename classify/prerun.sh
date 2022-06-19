apt-get update
apt-get install -y libgl1-mesa-dev
apt-get install -y libfreetype6-dev
pip uninstall -y pillow
pip install --no-cache-dir pillow