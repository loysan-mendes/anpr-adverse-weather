import http.client
import io
import json
import threading
import unittest
from unittest.mock import patch
from PIL import Image
from server import Handler, ThreadingHTTPServer, INFERENCE_LOCK, prepare_image


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, path, body=None, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port)
        conn.request('POST' if body is not None else 'GET', path, body, headers or {})
        response = conn.getresponse()
        value = response.read()
        conn.close()
        return response.status, value

    def test_static_and_path_isolation(self):
        self.assertEqual(self.request('/')[0], 200)
        self.assertEqual(self.request('/api/health')[0], 200)
        self.assertEqual(self.request('/../README.md')[0], 404)

    def test_rejects_invalid_upload_and_recovers(self):
        self.assertEqual(self.request('/api/analyze', b'not an image')[0], 400)
        self.assertFalse(INFERENCE_LOCK.locked())
        self.assertEqual(self.request('/api/analyze', b'')[0], 413)

    def test_rejects_cross_origin_and_foreign_host(self):
        self.assertEqual(self.request('/api/analyze', b'x', {'Origin': 'https://example.com'})[0], 403)
        self.assertEqual(self.request('/', headers={'Host': 'example.com'})[0], 403)

    def test_busy_and_invalid_profile(self):
        with INFERENCE_LOCK:
            self.assertEqual(self.request('/api/analyze', b'x')[0], 409)
        self.assertEqual(self.request('/api/analyze?profile=unknown', b'x')[0], 400)

    def test_upload_reaches_inference(self):
        buffer = io.BytesIO()
        Image.new('RGB', (20, 10)).save(buffer, format='PNG')
        with patch('server.infer', return_value={'plates': []}) as infer:
            status, body = self.request('/api/analyze?profile=exhaustive', buffer.getvalue())
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {'plates': []})
        self.assertEqual(infer.call_args.args[0].size, (20, 10))
        self.assertEqual(infer.call_args.args[1], 'exhaustive')

    def test_orientation_is_normalized(self):
        image = Image.new('RGB', (20, 10))
        exif = image.getexif()
        exif[274] = 6
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', exif=exif)
        self.assertEqual(prepare_image(buffer.getvalue()).size, (10, 20))


if __name__ == '__main__':
    unittest.main()
