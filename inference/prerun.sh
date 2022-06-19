apt-get update
apt-get install -y libmagic1
apt-get install -y libglib2.0-0
apt-get install -y ffmpeg libsm6 libxext6
pip uninstall -y pillow
pip install --no-cache-dir pillow