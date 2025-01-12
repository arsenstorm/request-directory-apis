# TikTok DL API

This API is based on the work found
[here](https://github.com/ytdl-org/youtube-dl).

It’s designed to be used with Request Directory and you can find more details
[here](https://request.directory/tiktok-dl).

## Development

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the API

```bash
python src/main.py
```

## Usage

By default, the API runs on port 7006.

```bash
docker run -it -p7006:7006 ghcr.io/arsenstorm/tiktok-dl:latest
```

## API

To use the API, you need to send a POST request containing JSON data to the
`/download` endpoint with the following parameters:

#### Parameters

- `url`: The URL of the video to download.

#### Example Request

As an example, we’ll use the following URL:

```bash
curl -X POST http://localhost:7006/download -H "Content-Type: application/json" -d '{"url": "https://www.tiktok.com/@tiktok/video/7268821288821821446"}'
```

#### Example Response

We get the following response:

```json
{
  "video_id": "7268821288821821446",
  "download_url": "https://request.directory/download/tiktok/...", // The downloaded video expires after 24 hours
  "expires_at": "Tue, 14 Jan 2025 22:23:11 GMT", // The expiration date of the downloaded video
  "success": true
}
```

In this response, we’ve received these details:

- `video_id`: The ID of the video.
- `download_url`: The URL of the downloaded video.
- `expires_at`: The expiration date of the downloaded video.
- `success`: Whether the request was successful.

## Notes

- You’ll need to export your cookies to host this API as TikTok actively blocks
  requests is suspects of scraping.
