# Youtube DL API

This API is based on the work found
[here](https://github.com/ytdl-org/youtube-dl).

It’s designed to be used with Request Directory and you can find more details
[here](https://request.directory/youtube-dl).

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

By default, the API runs on port 7005.

```bash
docker run -it -p7005:7005 ghcr.io/arsenstorm/youtube-dl:latest
```

## API

To use the API, you need to send a POST request containing JSON data to the
`/download` endpoint with the following parameters:

#### Parameters

- `url`: The URL of the video to download.

#### Example Request

As an example, we’ll use the following URL:

```bash
curl -X POST http://localhost:7005/download -H "Content-Type: application/json" -d '{"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

#### Example Response

We get the following response:

```json
{
  "result": {
    "video_id": "dQw4w9WgXcQ",
    "thumbnails": {
      "max": "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg", // 1280x720
      "high": "https://i.ytimg.com/vi/dQw4w9WgXcQ/sddefault.jpg", // 640x360
      "mid": "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg", // 480x360
      "low": "https://i.ytimg.com/vi/dQw4w9WgXcQ/mqdefault.jpg", // 320x180
      "min": "https://i.ytimg.com/vi/dQw4w9WgXcQ/default.jpg" // 120x90
    },
    "download_url": "https://request.directory/download/..." // The downloaded video expires after 24 hours
  },
  "success": true
}
```

In this response, we’ve received these details:

- `result`: The important stuff.
- `success`: Whether the request was successful.
