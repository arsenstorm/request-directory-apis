import os
from urllib.parse import urlparse
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import yt_dlp
from datetime import datetime, timedelta
import boto3
from botocore.client import Config
from pathlib import Path

load_dotenv()

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent
DOWNLOAD_DIR = PROJECT_ROOT / 'downloads'
DOWNLOAD_DIR.mkdir(exist_ok=True)


DEBUG_MODE = os.getenv('TIKTOKDL_DEBUG', 'false').lower() == 'true'
PORT = int(os.getenv('TIKTOKDL_PORT', '7006'))


COOKIE_FILE = PROJECT_ROOT / 'cookies.txt'
COOKIE_FILE_EXISTS = COOKIE_FILE.is_file()

if not COOKIE_FILE_EXISTS:
    print("Warning: cookies.txt file not found. Some videos may be inaccessible.")
elif os.path.getsize(COOKIE_FILE) == 0:
    print("Warning: cookies.txt file is empty. Some videos may be inaccessible.")


required_env_vars = ['R2_ENDPOINT', 'R2_ACCESS_KEY',
                     'R2_SECRET_KEY', 'R2_BUCKET_NAME', 'R2_PUBLIC_URL']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(
        f"Missing required environment variables: {', '.join(missing_vars)}")

s3 = boto3.client('s3',
                  endpoint_url=os.getenv('R2_ENDPOINT'),
                  aws_access_key_id=os.getenv('R2_ACCESS_KEY'),
                  aws_secret_access_key=os.getenv('R2_SECRET_KEY'),
                  config=Config(signature_version='s3v4'),
                  region_name='auto'
                  )

app = Flask(__name__)


def get_id_from_url(url):
    new_url = urlparse(url)
    if not new_url.hostname.endswith('tiktok.com'):
        return None
    return f"{new_url.hostname}/{new_url.path}".replace('//', '/')


@app.route('/download', methods=['POST'])
def download():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({
            "error": "You haven't included a URL in the request body.",
            "success": False
        }), 400

    url = data['url']
    video_url = get_id_from_url(url)
    print(video_url)

    if not video_url:
        return jsonify({
            "error": "Invalid TikTok URL.",
            "success": False
        }), 400

    try:
        ydl_opts = {
            'format': 'best',
            'outtmpl': str(DOWNLOAD_DIR / '%(id)s.%(ext)s'),
            'nocheckcertificate': True,
            'ignoreerrors': False,
            'quiet': False,
            'no_warnings': False,
            'extract_flat': False,
            'ssl_verify': False,
            'cookiefile': str(COOKIE_FILE) if COOKIE_FILE_EXISTS else None,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            video_id = video_url.split('/')[-1]
            info = ydl.extract_info(video_url, download=False)
            r2_key = f"tiktok/{video_id}.{info['ext']}"
            download_url = f"{os.getenv('R2_PUBLIC_URL')}/{r2_key}"

            try:
                if s3.head_object(Bucket=os.getenv('R2_BUCKET_NAME'), Key=r2_key):
                    return jsonify({
                        "video_id": video_id,
                        "download_url": download_url,
                        "expires_at": datetime.now() + timedelta(minutes=3600),
                        "success": True

                    })
            except s3.exceptions.ClientError as e:
                if e.response['Error']['Code'] != '404':
                    raise

            try:
                ydl.download([url])
            except Exception:
                return jsonify({
                    "error": "We’ve been unable to download this video from YouTube. Please try again later.",
                    "success": False
                }), 500

            try:
                local_file = str(DOWNLOAD_DIR / f"{video_id}.{info['ext']}")
                s3.upload_file(local_file, os.getenv('R2_BUCKET_NAME'), r2_key)
                os.remove(local_file)
            except Exception as e:
                print(e)
                return jsonify({
                    "error": "We’ve been unable to upload this video to storage. Please try again later.",
                    "success": False
                }), 500

            return jsonify({
                "video_id": video_id,
                "download_url": download_url,
                "expires_at": datetime.now() + timedelta(minutes=3600),
                "success": True
            })

    except Exception:
        return jsonify({
            "error": "We’ve been unable to download this video. Please try again later.",
            "success": False
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG_MODE)
