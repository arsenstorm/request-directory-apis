# Image Talker API

This API is based on the work found
[here](https://github.com/<insert-link>).

It’s designed to be used with Request Directory and you can find more details
[here](https://request.directory/imagetalker)

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

By default, the API runs on port 7004.

```bash
docker run -it -p7004:7004 ghcr.io/arsenstorm/imagetalker:latest
```

## API

To use the API, you need to send a POST request containing form data to the
`/create` endpoint with the following parameters:

#### Parameters

- `image`: The image.
- `audio`: The audio.

#### Example Request

As an example, we’ll...

```bash
curl -X POST http://localhost:7004/create -F "image=@.github/imagetalker/example_input.jpg"
```

#### Example Response

We get the following response:

```json
{
  // ...
  "result": {},
  "success": true
}
```

In this response, we’ve received these details:

- `result`: The important stuff.
- `success`: Whether the request was successful.
