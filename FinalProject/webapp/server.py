"""Run locally: python server.py. Submitted code is never executed or saved."""
import argparse
import json
import hashlib
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlsplit, parse_qs
import io
import zipfile
from inference import Engine, DISPLAY
from generation import available_models, generate
from study import StudyRunner, PROMPTS
from results import ResultStore

STATIC = Path(__file__).parent / 'static'
ENGINE = None
STUDY = None
RESULTS = None

def benchmark_samples():
    folder = Path(__file__).resolve().parents[1] / 'test_samples'
    manifest = json.loads((folder / 'sample_manifest.json').read_text(encoding='utf-8'))
    samples = []
    for entry in manifest:
        name = entry['file']
        if Path(name).name != name or not name.endswith('.c'):
            raise ValueError('Invalid benchmark filename.')
        content = (folder / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != entry['source_sha256']:
            raise ValueError('Benchmark sample hash mismatch.')
        samples.append({**entry, 'name': DISPLAY[entry['label']], 'code': content.decode('utf-8')})
    return samples

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Log routes and status only, never request bodies.
        super().log_message(fmt, *args)

    def reply(self, status, data, content_type='application/json; charset=utf-8'):
        payload = json.dumps(data).encode() if isinstance(data, dict) else data
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/api/results': return self.reply(200,RESULTS.read())
        if path == '/api/prompts':
            return self.reply(200, {'prompts':[{'task':task,'prompt':prompt} for task,prompt in PROMPTS]})
        if path == '/api/studies':
            runs=[]
            for file in sorted(STUDY.root.glob('*/study.json'),reverse=True):
                try:
                    state=STUDY.read(file.parent.name)
                    runs.append({'id':state['id'],'status':state['status'],'created_at':state['created_at']})
                except (OSError,ValueError,KeyError): pass
            return self.reply(200, {'runs':runs})
        if path in ['/api/study', '/api/study/download']:
            try:
                run_id=parse_qs(urlsplit(self.path).query).get('run_id',[''])[0]
                state=STUDY.read(run_id)
                if path == '/api/study': return self.reply(200,state)
                buffer=io.BytesIO()
                with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
                    folder=STUDY.folder(run_id)
                    for file in folder.iterdir():
                        if file.suffix in ['.json','.csv','.c']: archive.write(file,file.name)
                return self.reply(200,buffer.getvalue(),'application/zip')
            except (OSError,ValueError,KeyError): return self.reply(404,{'error':'Study not found.'})
        if path == '/api/status':
            return self.reply(200, {'ready': True, 'device': str(ENGINE.device), 'labels': DISPLAY})
        if path == '/api/generator-models':
            try: return self.reply(200, available_models())
            except Exception: return self.reply(502, {'error': 'Could not check OpenRouter availability. Try refreshing.'})
        if path == '/api/samples':
            try:
                return self.reply(200, {'samples': benchmark_samples()})
            except (OSError, ValueError, KeyError):
                return self.reply(503, {'error': 'Benchmark samples are unavailable. You can still paste code or open a file.'})
        files = {'/': ('home.html', 'text/html; charset=utf-8'),
                 '/live': ('index.html', 'text/html; charset=utf-8'),
                 '/results': ('results.html', 'text/html; charset=utf-8'),
                 '/results.js': ('results.js', 'text/javascript; charset=utf-8'),
                 '/generate': ('generate.html', 'text/html; charset=utf-8'),
                 '/study': ('study.html', 'text/html; charset=utf-8'),
                 '/study.js': ('study.js', 'text/javascript; charset=utf-8'),
                 '/generator.js': ('generator.js', 'text/javascript; charset=utf-8'),
                 '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                 '/style.css': ('style.css', 'text/css; charset=utf-8')}
        if path not in files: return self.reply(404, {'error': 'Not found.'})
        name, content_type = files[path]
        return self.reply(200, (STATIC / name).read_bytes(), content_type)

    def do_POST(self):
        route = urlsplit(self.path).path
        if route not in ['/api/analyze', '/api/generate','/api/study/start','/api/study/pause','/api/results/import']:
            return self.reply(404, {'error': 'Not found.'})
        # Accept only same-origin browser requests; the service binds to localhost.
        origin = self.headers.get('Origin')
        if origin and origin not in [f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}']:
            return self.reply(403, {'error': 'Use the local web page to submit code.'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= (28_000_000 if route=='/api/results/import' else 500000):
                return self.reply(413, {'error': 'The submitted code is too large or empty.'})
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict): raise ValueError('Expected a JSON object.')
            if route == '/api/results/import': result=RESULTS.upload(data)
            elif route == '/api/study/start': result=STUDY.start(data)
            elif route == '/api/study/pause':
                STUDY.pause_event.set();result={'status':'Pause requested; the current operation will finish and save.'}
            else: result = generate(data) if route == '/api/generate' else ENGINE.analyze(data.get('code'), data.get('model', '050'), data.get('strip', False),data.get('mode','first512'))
            return self.reply(200, result)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return self.reply(400, {'error': str(exc)})
        except Exception as exc:
            print(f'Request failed: {type(exc).__name__}', flush=True)
            return self.reply(500, {'error': 'The model could not finish. Check the server window or restart with --device cpu.'})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    ENGINE = Engine(args.device)
    STUDY = StudyRunner(ENGINE)
    RESULTS = ResultStore()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Open http://127.0.0.1:{args.port} | {args.device} | Ctrl+C to stop', flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
