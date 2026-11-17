"""Start the local demo and open its browser after it is ready."""
from pathlib import Path
import json, subprocess, sys, threading, time, urllib.request, webbrowser

ROOT=Path(__file__).resolve().parent
URL='http://127.0.0.1:8765'
def ready():
    try:
        with urllib.request.urlopen(URL+'/api/status',timeout=1) as response:
            status=json.load(response)
            return status.get('ready') and len(status.get('labels',{}))==5
    except Exception: return False

if ready():
    webbrowser.open(URL+'/')
    print('The platform is already running. Opened its project hub.')
else:
    child=subprocess.Popen([sys.executable,str(ROOT/'webapp/server.py'),'--device','cpu'],cwd=ROOT)
    def open_when_ready():
        for _ in range(120):
            if child.poll() is not None: return
            if ready():
                webbrowser.open(URL+'/');return
            time.sleep(1)
        print('Startup is taking longer. Check this window for errors.')
    threading.Thread(target=open_when_ready,daemon=True).start()
    print('Keep this window open. Press Ctrl+C to stop the platform.')
    try: child.wait()
    except KeyboardInterrupt:
        child.terminate();child.wait()
