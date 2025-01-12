import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import yt_dlp
from datetime import datetime, timedelta
import boto3
from botocore.client import Config
from pathlib import Path

load_dotenv()

DOWNLOAD_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / 'downloads'
DOWNLOAD_DIR.mkdir(exist_ok=True)
DEBUG_MODE = os.getenv('YOUTUBEDL_DEBUG', 'false').lower() == 'true'
PORT = int(os.getenv('YOUTUBEDL_PORT', '7005'))
COOKIE_FILE = Path(os.path.dirname(os.path.abspath(__file__))) / 'cookies.txt'
COOKIE_FILE_EXISTS = COOKIE_FILE.exists()
VALID_YOUTUBE_VIDEO_URLS = [
    'https://www.youtube.com/watch?v=',
    'https://youtu.be/',
    'https://m.youtube.com/watch?v=',
    'https://www.youtube.com/embed/',
    'https://www.youtube.com/v/',
    'https://www.youtube.com/shorts/',
    'https://www.youtube.com/live/',
    'https://music.youtube.com/watch?v=',
]

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
    for valid_url in VALID_YOUTUBE_VIDEO_URLS:
        if url.startswith(valid_url):
            return url.split(valid_url)[1]
    return None


@app.route('/download', methods=['POST'])
def download():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({
            "error": "You haven't included a URL in the request body.",
            "success": False
        }), 400

    url = data['url']
    video_id = get_id_from_url(url)
    print(video_id)

    if not video_id:
        return jsonify({
            "error": "Invalid YouTube URL.",
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
            info = ydl.extract_info(url, download=False)
            r2_key = f"{video_id}.{info['ext']}"
            download_url = f"{os.getenv('R2_PUBLIC_URL')}/{r2_key}"

            try:
                if s3.head_object(Bucket=os.getenv('R2_BUCKET_NAME'), Key=r2_key):
                    return jsonify({
                        "result": {
                            "video_id": video_id,
                            "thumbnails": {
                                "max": f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg",
                                "high": f"https://i.ytimg.com/vi/{video_id}/sddefault.jpg",
                                "mid": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
                                "low": f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg",
                                "min": f"https://i.ytimg.com/vi/{video_id}/default.jpg",
                            },
                            "download_url": download_url,
                            "expires_at": datetime.now() + timedelta(minutes=3600)
                        },
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
                "result": {
                    "video_id": video_id,
                    "thumbnails": {
                        "max": f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg",
                        "high": f"https://i.ytimg.com/vi/{video_id}/sddefault.jpg",
                        "mid": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
                        "low": f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg",
                        "min": f"https://i.ytimg.com/vi/{video_id}/default.jpg",
                    },
                    "download_url": download_url,
                    "expires_at": datetime.now() + timedelta(minutes=3600)
                },
                "success": True
            })

    except Exception:
        return jsonify({
            "error": "We’ve been unable to download this video. Please try again later.",
            "success": False
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG_MODE)
