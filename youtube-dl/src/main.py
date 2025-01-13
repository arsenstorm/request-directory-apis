import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import yt_dlp
from datetime import datetime, timedelta
import boto3
from botocore.client import Config
from pathlib import Path
import math
from concurrent.futures import ThreadPoolExecutor
import html

load_dotenv()

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent
DOWNLOAD_DIR = PROJECT_ROOT / 'downloads'
DOWNLOAD_DIR.mkdir(exist_ok=True)

DEBUG_MODE = os.getenv('YOUTUBEDL_DEBUG', 'false').lower() == 'true'
PORT = int(os.getenv('YOUTUBEDL_PORT', '7005'))

COOKIE_FILE = PROJECT_ROOT / 'cookies.txt'
COOKIE_FILE_EXISTS = COOKIE_FILE.is_file()

required_env_vars = ['R2_ENDPOINT', 'R2_ACCESS_KEY',
                     'R2_SECRET_KEY', 'R2_BUCKET_NAME', 'R2_PUBLIC_URL']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {
                     ', '.join(missing_vars)}")

s3 = boto3.client('s3',
                  endpoint_url=os.getenv('R2_ENDPOINT'),
                  aws_access_key_id=os.getenv('R2_ACCESS_KEY'),
                  aws_secret_access_key=os.getenv('R2_SECRET_KEY'),
                  config=Config(signature_version='s3v4'),
                  region_name='auto')

app = Flask(__name__)


def get_id_from_url(url):
    return url.split("v=")[-1].split("&")[0] if "v=" in url else url.split("/")[-1]


def process_subtitle(subtitle_file):
    subtitles = []

    with open(subtitle_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if "<p" in line:
                start = line.split("begin=")[1].split("\"")[1]
                text = line.split(">")[1].split("</p")[0]
                cleaned_text = html.unescape(text)
                subtitles.append(f"{start}: {cleaned_text}")

    return '\n'.join(subtitles)


def save_text_file(content, filename):
    file_path = DOWNLOAD_DIR / filename
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return file_path


def upload_multipart(local_file, r2_key):
    file_size = os.path.getsize(str(local_file))

    chunk_size = min(
        max(math.ceil(file_size / 10000), 10 * 1024 * 1024),
        100 * 1024 * 1024
    )

    multipart = s3.create_multipart_upload(
        Bucket=os.getenv('R2_BUCKET_NAME'),
        Key=r2_key
    )

    parts = []
    threads = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        with open(str(local_file), 'rb') as f:
            part_number = 1
            while True:
                data = f.read(chunk_size)
                if not data:
                    break

                future = executor.submit(
                    s3.upload_part,
                    Bucket=os.getenv('R2_BUCKET_NAME'),
                    Key=r2_key,
                    PartNumber=part_number,
                    UploadId=multipart['UploadId'],
                    Body=data
                )
                threads.append((part_number, future))
                part_number += 1

    for part_number, future in threads:
        result = future.result()
        parts.append({
            'PartNumber': part_number,
            'ETag': result['ETag']
        })

    s3.complete_multipart_upload(
        Bucket=os.getenv('R2_BUCKET_NAME'),
        Key=r2_key,
        UploadId=multipart['UploadId'],
        MultipartUpload={'Parts': sorted(parts, key=lambda x: x['PartNumber'])}
    )
    return f"{os.getenv('R2_PUBLIC_URL')}/{r2_key}"


def upload_to_r2(local_file, r2_key):
    file_size = os.path.getsize(str(local_file))

    if file_size < 100 * 1024 * 1024:
        s3.upload_file(local_file, os.getenv('R2_BUCKET_NAME'), r2_key)
        return f"{os.getenv('R2_PUBLIC_URL')}/{r2_key}"

    return upload_multipart(local_file, r2_key)


def get_r2_url(file_name):
    return f"{os.getenv('R2_PUBLIC_URL')}/youtube/{file_name}"


@app.route('/download', methods=['POST'])
def download():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "No URL provided.", "success": False}), 400

    url = data['url']
    video_id = get_id_from_url(url)

    if not video_id:
        return jsonify({"error": "Invalid YouTube URL.", "success": False}), 400

    try:
        ydl_opts = {
            'format': 'bestvideo[ext=mp4][vcodec^=avc][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][vcodec^=avc][height<=1080]/best[ext=mp4][height<=1080]',
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
            'outtmpl': {
                'default': str(DOWNLOAD_DIR / '%(id)s.%(ext)s'),
                'subtitle': str(DOWNLOAD_DIR / '%(id)s.%(ext)s')
            },
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'subtitlesformat': 'ttml',
            'nocheckcertificate': True,
            # 'quiet': True,
            # 'no_warnings': True,
            'cookiefile': str(COOKIE_FILE) if COOKIE_FILE_EXISTS else None,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            video_url = get_r2_url(f"{video_id}.mp4")
            metadata_url = get_r2_url(f"{video_id}_metadata.txt")
            subtitle_url = get_r2_url(f"{video_id}_subtitles.txt")

            # See if video is available
            try:
                if s3.head_object(Bucket=os.getenv('R2_BUCKET_NAME'), Key=f"youtube/{video_id}.mp4"):
                    return jsonify({
                        "video_id": video_id,
                        "video_url": video_url,
                        "metadata_url": metadata_url,
                        "subtitles_url": subtitle_url,
                        "thumbnails": {
                            "max": f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg",
                            "high": f"https://i.ytimg.com/vi/{video_id}/sddefault.jpg",
                            "mid": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
                            "low": f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg",
                            "min": f"https://i.ytimg.com/vi/{video_id}/default.jpg",
                        },
                        "expires_at": (datetime.now() + timedelta(minutes=3600)).isoformat(),
                        "success": True
                    })
            except s3.exceptions.ClientError as e:
                if e.response['Error']['Code'] != '404':
                    raise

            # Download video
            ydl.download([url])

            # Extract title and description
            title = info.get('title', 'No Title')
            description = info.get('description', 'No Description')

            subtitles = "No subtitles available."
            subtitle_file = DOWNLOAD_DIR / f"{video_id}.en.ttml"

            if subtitle_file.is_file():
                subtitles = process_subtitle(subtitle_file)

            subtitle_file = save_text_file(
                subtitles, f"{video_id}_subtitles.txt")
            print(subtitles[:300])

            # Save metadata
            metadata_content = f"title:{title}\ndescription:{description}"
            # Formatted as:
            # title:[TITLE]
            # description:[DESCRIPTION]
            metadata_file = save_text_file(
                metadata_content, f"{video_id}_metadata.txt")

            # Upload metadata
            print("INFO: Uploading metadata...")
            upload_to_r2(
                metadata_file, f"youtube/{video_id}_metadata.txt")
            print("INFO: Metadata uploaded successfully.")

            # Upload subtitles
            print("INFO: Uploading subtitles...")
            upload_to_r2(
                subtitle_file, f"youtube/{video_id}_subtitles.txt")
            print("INFO: Subtitles uploaded successfully.")

            # Upload video
            print("INFO: Uploading video...")
            video_file = DOWNLOAD_DIR / f"{video_id}.mp4"
            upload_to_r2(video_file, f"youtube/{video_id}.mp4")
            print("INFO: Video uploaded successfully.")

            # Cleanup
            os.remove(metadata_file)
            os.remove(video_file)
            os.remove(subtitle_file)

            return jsonify({
                "video_id": video_id,
                "video_url": video_url,
                "metadata_url": metadata_url,
                "subtitles_url": subtitle_url,
                "thumbnails": {
                    "max": f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg",
                    "high": f"https://i.ytimg.com/vi/{video_id}/sddefault.jpg",
                    "mid": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
                    "low": f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg",
                    "min": f"https://i.ytimg.com/vi/{video_id}/default.jpg",
                },
                "expires_at": (datetime.now() + timedelta(minutes=3600)).isoformat(),
                "success": True
            })

    except Exception as e:
        print(e)
        return jsonify({"error": "Failed to process the video.", "success": False}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG_MODE)
