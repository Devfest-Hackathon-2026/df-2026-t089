"""Read a product photo with the Cloud Vision API the same way the app does, to check the key and the reading.

Every run is one billed request (the free tier is 1000 a month), so it reads one photo per run.

  python tools/vision_ocr_test.py tools/assets/item1.jpg
"""
import base64, io, json, sys, urllib.error, urllib.request
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cloud_test as ct

URL = "https://vision.googleapis.com/v1/images:annotate"
LONG_SIDE = 1920  # StopLabeler.OCR_MAX_LONG_SIDE


def lines(response, image_height):
    """Same as CloudVisionParser.lines: split at line breaks, height = median symbol height."""
    out, text, heights, top = [], "", [], None
    def end():
        nonlocal text, heights, top
        if text.strip() and heights:
            out.append((text.strip(), sorted(heights)[len(heights) // 2] / image_height, top / image_height))
        text, heights, top = "", [], None
    for page in response.get("fullTextAnnotation", {}).get("pages", []):
        for block in page.get("blocks", []):
            for para in block.get("paragraphs", []):
                for word in para.get("words", []):
                    for s in word.get("symbols", []):
                        text += s.get("text", "")
                        ys = [max(0, v.get("y", 0)) for v in s.get("boundingBox", {}).get("vertices", [])]
                        if ys:
                            heights.append(max(ys) - min(ys))
                            top = min(ys) if top is None else min(top, min(ys))
                        brk = s.get("property", {}).get("detectedBreak", {}).get("type")
                        if brk in ("SPACE", "SURE_SPACE"):
                            text += " "
                        elif brk in ("EOL_SURE_SPACE", "LINE_BREAK"):
                            end()
    end()
    return out


def main():
    key = ct.local_property("CLOUD_VISION_API_KEY") or ct.local_property("GCP_API_KEY")
    if not key:
        sys.exit("no CLOUD_VISION_API_KEY or GCP_API_KEY in local.properties")
    im = ImageOps.exif_transpose(Image.open(sys.argv[1])).convert("RGB")
    im.thumbnail((LONG_SIDE, LONG_SIDE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    body = {"requests": [{"image": {"content": base64.b64encode(buf.getvalue()).decode()},
                          "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                          "imageContext": {"languageHints": ["ja"]}}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"x-goog-api-key": key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            res = json.load(r)["responses"][0]
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode()[:500]}")
    if "error" in res:
        sys.exit(f"error: {res['error']}")
    ls = lines(res, im.height)
    for t, h, top in sorted(ls, key=lambda x: -x[1])[:12]:
        print(f"height {h:.3f} top {top:.2f}  {t}")
    print("largest line:", max(ls, key=lambda x: x[1])[0] if ls else None)


if __name__ == "__main__":
    main()
