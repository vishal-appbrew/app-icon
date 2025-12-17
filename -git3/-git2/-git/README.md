# App Icon Generator

Generate beautiful 512x512 PNG app icons from brand websites automatically.

## Features

- **🎨 Beautiful Web Interface** — Easy-to-use frontend to generate icons with one click
- **Robust logo discovery** from `<img>`, `<link>`, meta tags, schema.org JSON-LD, or [Clearbit](https://logo.clearbit.com/) fallback
- **512x512 white background** with logo centered and minimum 128px padding (logo fits in 256x256)
- **Smart trimming** removes extra whitespace and transparent/pure-white edges  
- **No native dependencies** — deploys anywhere (Vercel, Render, etc.)
- **Pure Python** — uses only Pillow, BeautifulSoup, httpx, FastAPI

---

## Quickstart (Local Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

**🎉 Open your browser** and visit `http://127.0.0.1:8000` to use the web interface!

---

## Usage: Example cURL

```bash
curl -X POST http://127.0.0.1:8000/generate_icon \
  -H "Content-Type: application/json" \
  -d '{"website_url":"https://apple.com"}' \
  --output app_icon.png
```

**Response:** PNG image (512x512) with white background and centered logo

---

## API Endpoints

### `POST /generate_icon`

Generate an app icon from a website URL.

**Request Body:**
```json
{
  "website_url": "https://example.com"
}
```

**Response:**
- **200 OK**: PNG image (image/png)
- **400 Bad Request**: Missing or invalid `website_url`
- **404 Not Found**: Could not find a usable logo for the website

**Error Response Format:**
```json
{
  "error": "Error message here"
}
```

### `GET /`

**Web Interface** — Beautiful frontend to generate app icons. Visit this endpoint in your browser to use the interactive UI.

Features:
- Input URL and generate icon with one click
- Preview generated icon
- Download icon directly
- Example links for quick testing

### `GET /health`

Simple health check for monitoring.

---

## Deploy on Vercel

1. **Install Vercel CLI** (if not already installed):
   ```bash
   npm install -g vercel
   # or
   npx vercel
   ```

2. **Navigate to project directory:**
   ```bash
   cd app-icon-generator
   ```

3. **Deploy:**
   ```bash
   vercel
   # Follow prompts to link your project
   # For production: vercel --prod
   ```

4. **That's it!** Vercel will automatically:
   - Detect Python and install dependencies from `requirements.txt`
   - Build and deploy your FastAPI app
   - Route all requests to `main.py` via `vercel.json`

5. **Visit your deployment URL** to use the web interface!

**Note:** The `vercel.json` configuration routes all requests to `main.py`, which is handled by Vercel's Python runtime. The root endpoint (`/`) serves a beautiful web interface, while `/generate_icon` is the API endpoint.

---

## Deploy on Render.com

1. **Create a new Web Service** on [Render Dashboard](https://dashboard.render.com/new/web)

2. **Connect your Git repository** containing this project

3. **Configure the service:**
   - **Environment**: Python 3.9 or later
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Port**: Render will set `$PORT` automatically (usually 10000)

4. **Deploy!** Render will build and deploy your service automatically.

---

## Using with Custom GPT (Actions)

To integrate this API with OpenAI Custom GPT Actions:

1. **Add a new Action** in your Custom GPT configuration

2. **Configure the Action:**
   - **Method**: `POST`
   - **URL**: `https://your-deployment-url.com/generate_icon`
   - **Headers**: `Content-Type: application/json`
   - **Request Body Schema**:
     ```json
     {
       "type": "object",
       "properties": {
         "website_url": {
           "type": "string",
           "description": "The website URL to generate an app icon from"
         }
       },
       "required": ["website_url"]
     }
     ```
   - **Response**: `image/png` (the generated app icon)

3. **Test the integration** by providing a website URL through your Custom GPT

---

## How It Works

1. **Logo Discovery**: The service scrapes the website HTML for logo candidates in this order:
   - `<img>` tags with "logo", "brand", or "header" in alt/class/id
   - `<link rel="apple-touch-icon">`, `<link rel="icon">`, etc.
   - OpenGraph meta tags (`og:image`)
   - Twitter Card meta tags (`twitter:image`)
   - schema.org JSON-LD (`Organization.logo`, `ImageObject`)

2. **Candidate Scoring**: Candidates are scored by:
   - File type (SVG preferred, then PNG, then others)
   - URL contains "logo" keyword
   - Image dimensions (larger preferred)

3. **Image Processing**:
   - Downloads the best-scoring raster image (skips SVG)
   - Trims transparent and near-white borders
   - Resizes to fit in 256x256 box (preserving aspect ratio)
   - Centers on 512x512 white canvas

4. **Fallback**: If no usable logo is found, uses Clearbit Logo API

---

## Troubleshooting

### 400 Bad Request
- **Cause**: Missing or malformed `website_url`
- **Solution**: Ensure the request body includes a valid `website_url` field with an http/https URL

### 404 Not Found
- **Cause**: Could not find a usable logo for the website
- **Solution**: The website may not have discoverable logos, and Clearbit fallback also failed. Try a different website or check if the site is accessible.

### Timeout Issues
- **Cause**: Website is slow to respond or image download times out
- **Solution**: The service uses 10s timeout for HTML and 5s connect timeout for images. Very slow sites may fail.

### SVG Files
- **Note**: SVG files are skipped (not rasterized) to keep the service serverless-friendly. The service will fall back to Clearbit if only SVG logos are found.

---

## Tech Stack

- **FastAPI**: Modern, fast web framework
- **httpx**: Async HTTP client
- **Pillow**: Image processing
- **BeautifulSoup4 + lxml**: HTML parsing
- **pydantic**: Data validation

All dependencies are pure Python with no native extensions, making deployment easy on serverless platforms.

---

## License

MIT

