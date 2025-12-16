import io
import re
import logging
from typing import List, Dict, Optional, Tuple, Any
from urllib.parse import urljoin, urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel
from PIL import Image, ImageChops
from bs4 import BeautifulSoup, Tag
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app-icon-generator")

try:
    import cairosvg
except OSError:
    cairosvg = None
    logger.warning("CairoSVG not available - SVG conversion will fail")

app = FastAPI()


class GenerateIconRequest(BaseModel):
    website_url: str


# ---- Utility Functions ----

async def fetch_html(url: str, timeout: float = 10.0) -> Optional[str]:
    """Fetch raw HTML for a given URL asynchronously."""
    try:
        async with httpx.AsyncClient(
            timeout=timeout, 
            follow_redirects=True, 
            headers={"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"}
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            logger.info(f"Fetched HTML from {url} ({len(resp.text)} bytes)")
            return resp.text
    except Exception as e:
        logger.warning(f"Failed to fetch HTML from {url}: {e}")
        return None


def extract_candidates(html: str, base_url: str) -> List[Dict[str, Any]]:
    """
    Extract logo/image candidates from HTML using:
    - <img> with relevant alt/class/id
    - <link rel="apple-touch-icon"/mask-icon/icon/shortcut icon">
    - og: or twitter/meta images
    - schema.org logo/ImageObject via JSON-LD
    """
    soup = BeautifulSoup(html, "lxml")
    candidates = []

    def add_candidate(attrs: dict, label: str):
        """Helper to add a candidate to the list."""
        url = attrs.get('url')
        svg_content = attrs.get('svg_content') # Add support for inline SVG
        
        if not url and not svg_content:
            return
            
        candidates.append({
            'url': url,
            'svg_content': svg_content, # Store inline SVG content
            'type': attrs.get('type'),
            'label': label,
            'element': attrs.get('element'),
            'width': attrs.get('width'),
            'height': attrs.get('height'),
            'mime': attrs.get('mime'),
            'raw': attrs.get('raw'),
        })

    # Limit scope to header/nav if possible, but keep fallback
    # Strategy: Find candidates in header/nav first with high priority label
    # Then find elsewhere
    
    # Expanded header definition: tags OR classes containing 'header'
    header_tags = soup.find_all(['header', 'nav'])
    
    # Find divs (or any tag) with class containing 'header'
    # .layout--header, .site-header, etc.
    header_classes = soup.find_all(class_=lambda c: c and 'header' in str(c).lower())
    
    # Combine and dedupe
    all_header_containers = set(header_tags + header_classes)
    
    def process_container(container, context_label=""):
        # 1. <img> tags
        for img in container.find_all("img"):
            combined = " ".join([
                str(img.get("alt") or ""),
                str(img.get("class") or ""),
                str(img.get("id") or "")
            ]).lower()
            
            # Looser heuristic for header images - assume images in header are important
            is_header = "header" in context_label
            if is_header or any(k in combined for k in ["logo", "brand", "header"]):
                src = img.get("src") or img.get("data-src") or img.get("data-lazy-src") or ""
                srcset = img.get("srcset")

                if srcset:
                    srcset_entries = [s.strip().split()[0] for s in srcset.split(",") if s.strip()]
                    if srcset_entries:
                        src = srcset_entries[-1]

                resolved_url = resolve_url(src, base_url)
                if resolved_url:
                    width = try_int(img.get("width"))
                    height = try_int(img.get("height"))
                    add_candidate({
                        "url": resolved_url, 
                        "type": "img", 
                        "width": width, 
                        "height": height, 
                        "element": "img"
                    }, f"{context_label}img-tag")

        # 2. Inline <svg> tags
        for svg in container.find_all("svg"):
            # Check if it looks like a logo
            combined = " ".join([
                str(svg.get("class") or ""),
                str(svg.get("id") or ""),
                str(svg.get("aria-label") or "")
            ]).lower()
            
            # Check for title tag inside SVG for keywords
            title_tag = svg.find('title')
            if title_tag:
                 combined += " " + str(title_tag.string or "").lower()

            is_logo = any(k in combined for k in ["logo", "brand"])
            # If strictly logo-ish, add to label for scoring boost
            suffix = "-logo" if is_logo else ""
            
            if "header" in context_label or is_logo:
                # Serialize SVG
                svg_str = str(svg)
                width = try_int(svg.get("width"))
                height = try_int(svg.get("height"))
                
                # Check viewBox if width/height missing
                if not width and svg.get("viewBox"):
                    try:
                        parts = svg.get("viewBox").replace(",", " ").split()
                        if len(parts) == 4:
                            width = float(parts[2])
                            height = float(parts[3])
                    except:
                        pass

                add_candidate({
                    "svg_content": svg_str,
                    "type": "svg-inline",
                    "width": width,
                    "height": height,
                    "element": "svg"
                }, f"{context_label}svg-inline")

    # Process headers first
    if all_header_containers:
        for hdr in all_header_containers:
            process_container(hdr, "header-")
            
    # Process body (excluding already processed headers to avoid dupe logic? 
    # Actually duplicates are handled by de-duplication in scoring, so simple traversal is fine)
    # But we want to capture things NOT in header too
    process_container(soup, "body-")

    # 3. <link rel=...> for icons (Head only)
    rel_types = [
        ("apple-touch-icon", "apple"),
        ("mask-icon", "mask"),
        ("icon", "icon"),
        ("shortcut icon", "shortcut"),
    ]
    for rel, label in rel_types:
        for link in soup.find_all("link", rel=re.compile(rel, re.I)):
            href = link.get("href")
            if href:
                resolved_url = resolve_url(href, base_url)
                if resolved_url:
                    sizes = link.get("sizes")
                    width, height = None, None
                    if sizes:
                        parts = sizes.split("x")
                        if len(parts) == 2:
                            width = try_int(parts[0])
                            height = try_int(parts[1])
                    add_candidate({
                        "url": resolved_url, 
                        "type": "link", 
                        "width": width, 
                        "height": height, 
                        "element": "link"
                    }, f"link-rel:{rel}")

    # 4. OpenGraph/Twitter meta tags
    meta_properties = [
        ("property", "og:image", "og"),
        ("property", "og:logo", "og"),
        ("name", "twitter:image", "twitter"),
        ("name", "twitter:image:src", "twitter"),
    ]
    for attr, val, label in meta_properties:
        for meta in soup.find_all("meta", attrs={attr: val}):
            content = meta.get("content")
            if content:
                resolved_url = resolve_url(content, base_url)
                if resolved_url:
                    add_candidate({
                        "url": resolved_url, 
                        "type": "meta", 
                        "element": "meta"
                    }, f"meta-{label}")

    # 5. schema.org JSON-LD
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                continue
                
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                logo_url = None
                if "logo" in entry:
                    logo_url = entry["logo"]
                elif "image" in entry:
                    img_val = entry["image"]
                    if isinstance(img_val, str):
                        logo_url = img_val
                    elif isinstance(img_val, dict) and "url" in img_val:
                        logo_url = img_val["url"]
                
                if logo_url and isinstance(logo_url, str):
                    resolved_url = resolve_url(logo_url, base_url)
                    if resolved_url:
                        add_candidate({
                            "url": resolved_url, 
                            "type": "jsonld", 
                            "element": "ld"
                        }, "jsonld-logo")
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"Failed to parse JSON-LD: {e}")
            continue

    return candidates


def try_int(val):
    """Convert to int or None, handling 'px' suffix."""
    if not val:
        return None
    try:
        # Remove common units
        clean = str(val).lower().replace("px", "").replace("rem", "").strip()
        return int(float(clean))
    except (ValueError, TypeError):
        return None


def resolve_url(candidate_url: str, base_url: str) -> Optional[str]:
    """Ensure valid absolute URL for image resources."""
    if not candidate_url:
        return None
    candidate_url = candidate_url.strip()
    # Handle protocol-relative URLs
    if candidate_url.startswith("//"):
        return "https:" + candidate_url
    # Already absolute
    if candidate_url.startswith("http://") or candidate_url.startswith("https://"):
        return candidate_url
    # Relative URL - use urljoin
    return urljoin(base_url, candidate_url)


def score_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Score and sort logo candidates:
    - Prefer SVG, then PNG, then others
    - Prefer URLs with "logo" in them
    - Prefer larger dimensions
    - Deduplicate by final absolute URL
    """
    def ext_score(url: str, is_inline_svg: bool = False) -> int:
        if is_inline_svg:
            return 10 # Inline SVG is superior (vector)
            
        url_lower = url.lower()
        if url_lower.endswith(".svg"):
            return 3
        elif url_lower.endswith(".png"):
            return 2
        elif url_lower.endswith(".jpg") or url_lower.endswith(".jpeg"):
            return 1
        elif url_lower.endswith(".ico"):
            return 1
        else:
            return 0

    def logo_score(url: str, label: str) -> int:
        score = 0
        if "logo" in (url or "").lower(): score += 1
        if "logo" in label.lower(): score += 50 # Strong boost if we detected 'logo' in attributes
        if "header" in label: score += 500 # Massive boost for header items
        return score

    def size_score(width: Optional[int], height: Optional[int]) -> int:
        if width and height:
            return min(width, height)
        return 0

    scored = []
    seen = set()
    scored = []
    seen = set()
    for cand in candidates:
        abs_url = cand.get("url")
        svg_content = cand.get("svg_content")
        
        # Dedupe by URL if exists, or treat inline SVGs as unique per content hash (simplified)
        key = abs_url if abs_url else hash(svg_content)
        
        if key in seen:
            continue
        seen.add(key)
        
        is_inline = bool(svg_content)
        
        score = (
            ext_score(abs_url or "", is_inline) * 100 +
            logo_score(abs_url or "", cand.get('label', '')) * 25 +
            size_score(cand.get("width"), cand.get("height"))
        )
        cand['score'] = score
        scored.append(cand)
    
    # Sort by score descending
    return sorted(scored, key=lambda c: -c['score'])


async def fetch_image(
    url: str,
    timeout: float = 10.0,
    connect_timeout: float = 5.0,
    max_size: int = 10_000_000
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Download the image data and detect content-type.
    Uses headers first, then simple content sniffing.
    """
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=connect_timeout), 
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            data = resp.content
            
            if len(data) > max_size:
                logger.warning(f"Image from {url} too large ({len(data)} bytes)")
                return None, None
            
            # Content-type detection via sniffing if header is missing/unclear
            if not ctype or ctype == "application/octet-stream":
                sniff = data[:12]
                if sniff.startswith(b"\x89PNG\r\n\x1a\n"):
                    ctype = "image/png"
                elif sniff.startswith(b"\xFF\xD8\xFF"):
                    ctype = "image/jpeg"
                elif sniff.startswith(b"GIF87a") or sniff.startswith(b"GIF89a"):
                    ctype = "image/gif"
                elif b"<svg" in sniff or data.lstrip()[:100].startswith(b"<svg"):
                    ctype = "image/svg+xml"
            
            return data, ctype
    except Exception as e:
        logger.warning(f"Failed to download image {url}: {e}")
        return None, None


def is_svg(content_type: str, data: bytes, url: str) -> bool:
    """Determine if the image is SVG based on content-type, url extension or first bytes."""
    url_lower = url.lower() if url else ""
    content_type_lower = (content_type or "").lower()
    if "svg" in content_type_lower or url_lower.endswith(".svg"):
        return True
    # Check first bytes
    if data and (data.lstrip().startswith(b"<svg") or b"<svg" in data[:100]):
        return True
    return False


def convert_svg_to_png(svg_data: bytes, width: int = 1024, height: int = 1024) -> Optional[bytes]:
    """Convert SVG bytes to PNG bytes using Cairosvg."""
    if not cairosvg:
        logger.error("CairoSVG lib not loaded")
        return None
    try:
        return cairosvg.svg2png(bytestring=svg_data, output_width=width, output_height=height)
    except Exception as e:
        logger.warning(f"Failed to convert SVG to PNG: {e}")
        return None


def trim_whitespace(im: Image.Image, tolerance: int = 14) -> Image.Image:
    """
    Trim transparent or 'almost-white' edges from RGBA image using Pillow.
    Tolerance: how far from pure white (#FFF) to treat as still white (0-255 scale).
    """
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    
    # First, trim transparent edges
    bbox = im.getbbox()
    if bbox:
        im = im.crop(bbox)
    else:
        # Empty image
        return im

    # Next: remove near-white borders
    # Convert to array for easier processing
    width, height = im.size
    
    def is_whiteish(pixel: Tuple[int, int, int, int]) -> bool:
        r, g, b, a = pixel
        if a == 0:
            return True  # Transparent is treated as "white" for trimming
        # Check if all RGB values are within tolerance of 255
        return all(255 - c <= tolerance for c in (r, g, b))

    # Find crop bounds by checking edges
    # Top edge
    top = 0
    for y in range(height):
        if not all(is_whiteish(im.getpixel((x, y))) for x in range(width)):
            top = y
            break
    else:
        # All rows are whiteish - return minimal image
        return Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    
    # Bottom edge
    bottom = height
    for y in range(height - 1, -1, -1):
        if not all(is_whiteish(im.getpixel((x, y))) for x in range(width)):
            bottom = y + 1
            break
    
    # Left edge
    left = 0
    for x in range(width):
        if not all(is_whiteish(im.getpixel((x, y))) for y in range(height)):
            left = x
            break
    
    # Right edge
    right = width
    for x in range(width - 1, -1, -1):
        if not all(is_whiteish(im.getpixel((x, y))) for y in range(height)):
            right = x + 1
            break
    
    # Crop to remove white borders
    if (left, top, right, bottom) != (0, 0, width, height):
        im = im.crop((left, top, right, bottom))
    
    return im


def compose_icon(im: Image.Image) -> Image.Image:
    """
    Fit trimmed image into 256x256 max box (preserving aspect ratio),
    then center on 512x512 white background.
    Ensures at least 128px padding on all sides.
    """
    target_logo_size = 256  # Logo fits in this box
    total_size = 512  # Final icon size
    min_padding = 128  # Minimum padding on each side

    # Convert to RGBA if needed
    im = im.convert("RGBA")
    w, h = im.size
    
    # Calculate scaling to fit in 256x256 while maintaining aspect ratio
    # But ensure we have at least 128px padding, so max logo size is 512 - 2*128 = 256
    ratio = min(target_logo_size / w, target_logo_size / h, 1.0)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    
    # Resize with high-quality resampling
    im = im.resize((new_w, new_h), resample=Image.LANCZOS)

    # Create white background (RGBA with alpha=255 for opaque white)
    bg = Image.new("RGBA", (total_size, total_size), (255, 255, 255, 255))
    
    # Center the logo
    paste_x = (total_size - new_w) // 2
    paste_y = (total_size - new_h) // 2
    bg.paste(im, (paste_x, paste_y), mask=im if im.mode == "RGBA" else None)
    
    return bg.convert("RGBA")


async def get_logo_image_from_website(url: str) -> Optional[Image.Image]:
    """
    Main pipeline: fetch HTML, extract logo candidates, download and process best image.
    Falls back to Clearbit if no usable raster image is found.
    """
    # 1. Fetch HTML
    html = await fetch_html(url)
    if not html:
        logger.warning(f"Could not fetch HTML from {url}")
        return None

    # 2. Extract all image/logo candidates
    candidates = extract_candidates(html, url)
    candidates = score_candidates(candidates)
    logger.info(f"Extracted {len(candidates)} candidates for {url}")

    # 3. Try to download best raster (non-SVG) candidates one by one
    # 3. Try to download best candidates one by one
    for cand in candidates:
        logger.info(f"Trying candidate: {cand.get('url', 'inline-svg')} (score: {cand.get('score', 0)})")
        
        img_data = None
        ctype = None
        
        # Case A: Inline SVG
        if cand.get("svg_content"):
            try:
                svg_bytes = cand["svg_content"].encode('utf-8')
                # Convert to PNG
                png_data = convert_svg_to_png(svg_bytes)
                if png_data:
                    img_data = png_data
                    ctype = "image/png"
            except Exception as e:
                logger.warning(f"Error converting inline SVG: {e}")
                continue
                
        # Case B: Standard URL
        elif cand.get("url"):
            img_url = cand['url']
            dl_data, dl_ctype = await fetch_image(img_url)
            
            if dl_data:
                # Check for SVG
                if is_svg(dl_ctype, dl_data, img_url):
                    logger.info(f"Candidate {img_url} is SVG, converting...")
                    png_data = convert_svg_to_png(dl_data)
                    if png_data:
                        img_data = png_data
                        ctype = "image/png"
                    else:
                        logger.warning("SVG conversion failed, skipping")
                        continue
                else:
                    img_data = dl_data
                    ctype = dl_ctype
        
        if not img_data:
            continue
            
        # Try opening in Pillow
        try:
            pil_im = Image.open(io.BytesIO(img_data)).convert("RGBA")
            # Trim whitespace/padding
            trimmed = trim_whitespace(pil_im, tolerance=14)
            # Compose final icon
            icon_img = compose_icon(trimmed)
            logger.info(f"Successfully composed icon from candidate")
            return icon_img
        except Exception as e:
            logger.warning(f"Error processing image: {e}")
            continue

    # 4. Fallback: Clearbit logo API
    domain = urlparse(url).netloc
    if domain:
        # Remove www. prefix for Clearbit
        domain = domain.replace("www.", "")
        clearbit_url = f"https://logo.clearbit.com/{domain}?format=png"
        logger.info(f"Using Clearbit fallback: {clearbit_url}")
        
        img_data, ctype = await fetch_image(clearbit_url)
        # Clearbit usually returns PNG, but verify
        if img_data and ctype:
            # If specifically SVG, convert it
            if is_svg(ctype, img_data, clearbit_url):
                 img_data = convert_svg_to_png(img_data)
                 
            if img_data:
                try:
                    pil_im = Image.open(io.BytesIO(img_data)).convert("RGBA")
                    trimmed = trim_whitespace(pil_im, tolerance=14)
                    icon_img = compose_icon(trimmed)
                    logger.info(f"Successfully composed icon from Clearbit for {domain}")
                    return icon_img
                except Exception as e:
                    logger.warning(f"Error processing Clearbit image for {domain}: {e}")
                    return None
    
    return None


# ---- FastAPI Routes ----

@app.post("/generate_icon")
async def generate_icon(req: GenerateIconRequest):
    """
    Generate a 512x512 PNG app icon from a brand website.
    
    Request body:
    {
        "website_url": "https://example.com"
    }
    
    Returns: PNG image (512x512) with white background and centered logo
    """
    if not req.website_url:
        raise HTTPException(status_code=400, detail="Missing website_url parameter")

    # Basic URL validation
    if not re.match(r"^https?://", req.website_url):
        raise HTTPException(
            status_code=400,
            detail="website_url must be a valid http or https URL"
        )

    logger.info(f"Generating icon for: {req.website_url}")
    icon_img = await get_logo_image_from_website(req.website_url)
    
    if icon_img is None:
        raise HTTPException(
            status_code=404,
            detail=f"Could not find a usable logo for {req.website_url}"
        )

    # Convert to PNG bytes
    buf = io.BytesIO()
    icon_img.save(buf, format="PNG")
    buf.seek(0)
    
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={'Content-Disposition': 'inline; filename="app_icon.png"'}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom exception handler to return JSON errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML page."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>App Icon Generator</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 40px;
            max-width: 600px;
            width: 100%;
        }
        
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 2rem;
            text-align: center;
        }
        
        .subtitle {
            color: #666;
            text-align: center;
            margin-bottom: 30px;
            font-size: 0.95rem;
        }
        
        .form-group {
            margin-bottom: 25px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 500;
        }
        
        input[type="url"] {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1rem;
            transition: border-color 0.3s;
        }
        
        input[type="url"]:focus {
            outline: none;
            border-color: #667eea;
        }
        
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        
        .loading.active {
            display: block;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            display: none;
            background: #fee;
            color: #c33;
            padding: 12px;
            border-radius: 8px;
            margin: 20px 0;
            border: 1px solid #fcc;
        }
        
        .error.active {
            display: block;
        }
        
        .result {
            display: none;
            margin-top: 30px;
            text-align: center;
        }
        
        .result.active {
            display: block;
        }
        
        .result-image {
            max-width: 100%;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            margin: 20px 0;
        }
        
        .download-btn {
            margin-top: 15px;
            background: #28a745;
            width: auto;
            padding: 10px 24px;
            display: inline-block;
        }
        
        .download-btn:hover {
            background: #218838;
            box-shadow: 0 10px 20px rgba(40, 167, 69, 0.4);
        }
        
        .examples {
            margin-top: 30px;
            padding-top: 30px;
            border-top: 1px solid #e0e0e0;
        }
        
        .examples h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1rem;
        }
        
        .example-links {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .example-link {
            padding: 6px 12px;
            background: #f5f5f5;
            border-radius: 6px;
            text-decoration: none;
            color: #667eea;
            font-size: 0.85rem;
            transition: background 0.2s;
        }
        
        .example-link:hover {
            background: #e8e8e8;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 App Icon Generator</h1>
        <p class="subtitle">Generate beautiful 512x512 app icons from any website</p>
        
        <form id="iconForm">
            <div class="form-group">
                <label for="websiteUrl">Website URL</label>
                <input 
                    type="url" 
                    id="websiteUrl" 
                    name="websiteUrl" 
                    placeholder="https://example.com" 
                    required
                >
            </div>
            
            <button type="submit" id="generateBtn">Generate Icon</button>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="margin-top: 10px; color: #666;">Generating your icon...</p>
        </div>
        
        <div class="error" id="error"></div>
        
        <div class="result" id="result">
            <h3>Generated Icon</h3>
            <img id="resultImage" class="result-image" alt="Generated app icon">
            <br>
            <a id="downloadLink" class="download-btn" download="app_icon.png">Download Icon</a>
        </div>
        
        <div class="examples">
            <h3>Try these examples:</h3>
            <div class="example-links">
                <a href="#" class="example-link" data-url="https://www.nike.com/in/">Nike</a>
                <a href="#" class="example-link" data-url="https://ta3swim.com/">TA3 Swim</a>
                <a href="#" class="example-link" data-url="https://foreverbeaumore.com/">Forever Beaumore</a>
                <a href="#" class="example-link" data-url="https://apple.com/">Apple</a>
                <a href="#" class="example-link" data-url="https://stripe.com/">Stripe</a>
            </div>
        </div>
    </div>
    
    <script>
        const form = document.getElementById('iconForm');
        const websiteUrlInput = document.getElementById('websiteUrl');
        const generateBtn = document.getElementById('generateBtn');
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const result = document.getElementById('result');
        const resultImage = document.getElementById('resultImage');
        const downloadLink = document.getElementById('downloadLink');
        const exampleLinks = document.querySelectorAll('.example-link');
        
        // Handle example links
        exampleLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const url = link.getAttribute('data-url');
                websiteUrlInput.value = url;
                form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
            });
        });
        
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const url = websiteUrlInput.value.trim();
            if (!url) {
                showError('Please enter a valid URL');
                return;
            }
            
            // Validate URL format
            try {
                new URL(url);
            } catch {
                showError('Please enter a valid URL (must start with http:// or https://)');
                return;
            }
            
            // Reset UI
            hideError();
            hideResult();
            showLoading();
            generateBtn.disabled = true;
            
            try {
                const response = await fetch('/generate_icon', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ website_url: url }),
                });
                
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({ error: 'Failed to generate icon' }));
                    throw new Error(errorData.error || `Error: ${response.status} ${response.statusText}`);
                }
                
                // Get image blob
                const blob = await response.blob();
                const imageUrl = URL.createObjectURL(blob);
                
                // Display result
                resultImage.src = imageUrl;
                downloadLink.href = imageUrl;
                showResult();
                
            } catch (err) {
                showError(err.message || 'Failed to generate icon. Please try again.');
            } finally {
                hideLoading();
                generateBtn.disabled = false;
            }
        });
        
        function showLoading() {
            loading.classList.add('active');
        }
        
        function hideLoading() {
            loading.classList.remove('active');
        }
        
        function showError(message) {
            error.textContent = message;
            error.classList.add('active');
        }
        
        function hideError() {
            error.classList.remove('active');
        }
        
        function showResult() {
            result.classList.add('active');
        }
        
        function hideResult() {
            result.classList.remove('active');
        }
    </script>
</body>
</html>
    """


@app.get("/health")
async def health():
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}

