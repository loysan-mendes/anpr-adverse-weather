"""Local ANPR web application. Start with: ./run_ml.ps1 backend/server.py"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import logging
from pathlib import Path
import sys
import tempfile
import threading
from urllib.parse import parse_qs, urlsplit
import uuid
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ml/scripts'))
MAX_BYTES = 12 * 1024 * 1024
MAX_VIDEO_BYTES = 50 * 1024 * 1024
MAX_PIXELS = 12_000_000
INFERENCE_LOCK = threading.Lock()
VIDEO_RESULTS_DIR = tempfile.mkdtemp(prefix='anpr-videos-')
STATIC = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'),
          '/style.css': ('style.css', 'text/css')}


def prepare_image(payload):
    from PIL import Image, ImageOps, UnidentifiedImageError
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as source:
                if source.format not in {'JPEG', 'PNG', 'WEBP'}:
                    raise ValueError('Please upload a JPEG, PNG or WebP image.')
                if source.width * source.height > MAX_PIXELS:
                    raise ValueError('Image is too large. Use an image up to 12 megapixels.')
                source.load()
                return ImageOps.exif_transpose(source).convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError('The image could not be decoded safely. Try another file.') from exc


def infer(image, profile):
    from decision_engine import process_image
    # A private temporary file bridges the existing image-path inference API.
    with tempfile.TemporaryDirectory(prefix='anpr-') as directory:
        path = Path(directory) / 'upload.png'
        image.save(path)
        result = process_image(path, profile=profile, use_upscale=False)
    result.pop('image', None)
    result.pop('detector_weights', None)
    result['image_size'] = {'width': image.width, 'height': image.height}
    return result


def infer_video(video_bytes, frame_stride=1, is_demo=False):
    from track_and_read_video import VideoANPRTracker
    video_id = uuid.uuid4().hex[:12]
    out_video_path = Path(VIDEO_RESULTS_DIR) / f'annotated_{video_id}.mp4'
    out_json_path = Path(VIDEO_RESULTS_DIR) / f'passages_{video_id}.json'

    if is_demo:
        in_video_path = ROOT / 'demo_vehicle_passage.mp4'
        if not in_video_path.is_file():
            raise FileNotFoundError('Demo video not found on server.')
    else:
        in_video_path = Path(VIDEO_RESULTS_DIR) / f'input_{video_id}.mp4'
        in_video_path.write_bytes(video_bytes)

    tracker = VideoANPRTracker(frame_stride=frame_stride)
    result = tracker.process_video(
        video_path=in_video_path,
        out_json=out_json_path,
        out_video=out_video_path,
    )
    result['video_stream_url'] = f'/api/video-stream/{video_id}'
    return result


class Handler(BaseHTTPRequestHandler):
    def valid_host(self):
        port = self.server.server_port
        if self.headers.get('Host') not in {f'127.0.0.1:{port}', f'localhost:{port}'}:
            self.reply(403, {'error': 'Use the local application address.'})
            return False
        return True

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def reply(self, status, payload, content_type='application/json'):
        body = json.dumps(payload, allow_nan=False).encode() if content_type == 'application/json' else payload
        self.send_response(status)
        self.send_header('Content-Type', content_type + '; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.valid_host():
            return
        path = urlsplit(self.path).path
        if path == '/api/health':
            return self.reply(200, {'status': 'ready', 'busy': INFERENCE_LOCK.locked()})
        if path == '/api/sample':
            sample = ROOT / 'ml/data/raw/number_plate_images_ocr/number_plate_images_ocr/dc_auto_image_000021_bMXgvtud5K.jpg'
            if sample.is_file():
                return self.reply(200, sample.read_bytes(), 'image/jpeg')
            return self.reply(404, {'error': 'The sample image is not installed. Choose your own image.'})
        if path == '/api/sample-video':
            sample_video = ROOT / 'demo_vehicle_passage.mp4'
            if sample_video.is_file():
                return self.reply(200, sample_video.read_bytes(), 'video/mp4')
            return self.reply(404, {'error': 'The demo video file is not present.'})
        if path.startswith('/api/video-stream/'):
            video_id = path[len('/api/video-stream/'):]
            if not video_id.isalnum() or len(video_id) > 32:
                return self.reply(400, {'error': 'Invalid video ID.'})
            stream_file = Path(VIDEO_RESULTS_DIR) / f'annotated_{video_id}.mp4'
            if stream_file.is_file():
                return self.reply(200, stream_file.read_bytes(), 'video/mp4')
            return self.reply(404, {'error': 'Video stream expired or not found.'})
        if path in STATIC:
            name, mime = STATIC[path]
            return self.reply(200, (ROOT / 'frontend' / name).read_bytes(), mime)
        self.reply(404, {'error': 'Not found.'})

    def do_POST(self):
        if not self.valid_host():
            return
        target = urlsplit(self.path)
        if target.path not in {'/api/analyze', '/api/analyze-video'}:
            return self.reply(404, {'error': 'Not found.'})
        # Accept browser uploads only from this app; never expose credentialed CORS.
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            return self.reply(403, {'error': 'Upload from this application only.'})
        if self.headers.get('Transfer-Encoding'):
            return self.reply(400, {'error': 'A fixed upload size is required.'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            length = 0

        # Route 1: Multi-frame Video Analysis
        if target.path == '/api/analyze-video':
            is_demo = parse_qs(target.query).get('demo', ['0'])[0] in {'1', 'true'}
            try:
                stride = int(parse_qs(target.query).get('stride', ['1'])[0])
            except ValueError:
                stride = 1
            stride = max(1, min(5, stride))

            if is_demo and length == 0:
                payload = b''
            else:
                if not 0 < length <= MAX_VIDEO_BYTES:
                    return self.reply(413, {'error': 'Upload a video between 1 byte and 50 MB.'})
                try:
                    payload = self.rfile.read(length)
                except (TimeoutError, OSError):
                    return self.reply(408, {'error': 'Upload timed out. Please try again.'})
                if len(payload) != length:
                    return self.reply(400, {'error': 'Upload was interrupted. Please try again.'})

            if not INFERENCE_LOCK.acquire(blocking=False):
                return self.reply(409, {'error': 'The model is processing another task. Please try again shortly.'})
            try:
                result = infer_video(payload, frame_stride=stride, is_demo=is_demo)
                self.reply(200, result)
            except Exception as exc:
                logging.exception('Video analysis failed')
                self.reply(500, {'error': f'Video processing error: {str(exc)}'})
            finally:
                INFERENCE_LOCK.release()
            return

        # Route 2: Single Image Analysis
        if not 0 < length <= MAX_BYTES:
            return self.reply(413, {'error': 'Upload an image between 1 byte and 12 MB.'})
        profile = parse_qs(target.query).get('profile', ['balanced'])[0]
        # Drain the bounded body before replying (browsers may otherwise see a
        # connection reset instead of the busy/validation response).
        try:
            payload = self.rfile.read(length)
        except (TimeoutError, OSError):
            return self.reply(408, {'error': 'Upload timed out. Please try again.'})
        if len(payload) != length:
            return self.reply(400, {'error': 'Upload was interrupted. Please try again.'})
        if profile not in {'balanced', 'exhaustive'}:
            return self.reply(400, {'error': 'Unknown inference profile.'})
        if not INFERENCE_LOCK.acquire(blocking=False):
            return self.reply(409, {'error': 'The model is processing another image. Please try again shortly.'})
        try:
            image = prepare_image(payload)
            result = infer(image, profile)
            self.reply(200, result)
        except ValueError as exc:
            self.reply(400, {'error': str(exc)})
        except Exception:
            logging.exception('Image analysis failed')
            self.reply(500, {'error': 'Analysis failed. Check the server terminal for model or dependency details.'})
        finally:
            INFERENCE_LOCK.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.daemon_threads = True
    print(f'ANPR is running at http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
