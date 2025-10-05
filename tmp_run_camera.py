import time
from dash_app.scripts.camera_worker import start_camera, stop_camera
print('Starting camera for 12s...')
start_camera(cfg_path='config.json', voice_lang='en')
try:
    time.sleep(12)
finally:
    stop_camera()
    print('Stopped camera')
