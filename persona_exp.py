"""
Persona Ad Gen - Streamlit (Fully Autonomous, Gemini + Unsplash, 10+ Banners with Offers)
"""

import os
import io
import time
import random
from typing import Optional, List, Tuple
from PIL import Image, ImageDraw, ImageFont
import requests
import streamlit as st
from dotenv import load_dotenv

# Load .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# -------------------- Config --------------------
BANNER_SIZES = {
    "Instagram Post": (1080, 1080),
    "Facebook Ad": (1200, 628),
    "Twitter Post": (1600, 900),
    "LinkedIn Ad": (1200, 627),
    "Custom": (1280, 720)
}

COLOR_SCHEMES = {
    "Cool Blue": (0, 60, 120),
    "Forest Green": (34, 139, 34),
    "Royal Purple": (102, 51, 153),
    "Crimson": (220, 20, 60),
    "Charcoal": (54, 69, 79)
}

UNSPLASH_ACCESS_KEY = "your unsplash api key"  # Replace with your Unsplash key

CTA_OPTIONS = ["Shop Now", "Learn More", "Sign Up Today", "Get Started"]

OFFER_LINES = [
    "Limited Time Offer!",
    "Get 20% Off Today!",
    "Exclusive Deal for You!",
    "Don't Miss Out!",
    "Hurry, Offer Ends Soon!"
]

# Fonts list (add paths to different TTF fonts you have)
FONTS = ["arialbd.ttf", "arial.ttf"]

# -------------------- Utility Functions --------------------
def pil_from_bytes(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    except Exception as e:
        st.error(f"Failed to process image: {str(e)}")
        return None

def image_to_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

def fetch_unsplash_image(query: str, width: int, height: int) -> Optional[Image.Image]:
    """Fetch random image from Unsplash API."""
    try:
        url = f"https://api.unsplash.com/photos/random?query={query}&orientation=landscape&client_id={UNSPLASH_ACCESS_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()
        img_url = data.get("urls", {}).get("regular")
        if img_url:
            img_data = requests.get(img_url).content
            img = Image.open(io.BytesIO(img_data)).convert("RGBA")
            img.thumbnail((width, height))
            return img
        return None
    except Exception as e:
        st.warning(f"Unsplash fetch failed: {str(e)}")
        return None

# -------------------- AI Headline Generation --------------------
def generate_local_headlines(description: str, num: int = 15) -> List[str]:
    templates = [
        "Discover {desc} Like Never Before!",
        "Transform Your Life with {desc}!",
        "Why {desc} is the Secret to Success?",
        "Get Ready to Experience {desc} Today!",
        "The Ultimate Guide to {desc}!"
    ]
    headlines = []
    for _ in range(num):
        t = random.choice(templates)
        word = description.split()[0] if description else "This"
        headlines.append(t.format(desc=word))
    return headlines

def generate_gemini_headlines(description: str) -> List[str]:
    """Generate headlines via Gemini API if available."""
    try:
        if not GEMINI_API_KEY:
            return generate_local_headlines(description)
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"""
        Generate 10 catchy, unique ad headlines for this product/audience:
        "{description}"
        Rules:
        - Max 60 characters per headline
        - Use different styles (statement, question, command)
        - Include offers or urgency phrases if possible
        Return only the headlines, one per line.
        """
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        text = getattr(response, 'text', str(response))
        headlines = [line.strip('"').strip() for line in text.split('\n') if line.strip()]
        if len(headlines) < 5:
            return generate_local_headlines(description)
        return headlines[:10]
    except Exception as e:
        st.warning(f"Gemini API failed, using local headlines: {str(e)}")
        return generate_local_headlines(description)

# -------------------- Banner Composition --------------------
def draw_text_multiline(draw, text, font, max_width, x, y, shadow=True, line_spacing=4):
    lines = []
    words = text.split()
    line = ""
    for word in words:
        test_line = f"{line} {word}".strip()
        if draw.textlength(test_line, font=font) <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)

    bbox = font.getbbox("A")
    line_height = bbox[3] - bbox[1] + line_spacing

    for i, line in enumerate(lines):
        line_y = y + i * line_height
        if shadow:
            for dx, dy in [(-2, -2), (2, 2)]:
                draw.text((x + dx, line_y + dy), line, font=font, fill=(0,0,0,160))
        draw.text((x, line_y), line, font=font, fill=(255,255,255,255))
    return y + len(lines) * line_height

def compose_banner(
    scene_img: Image.Image,
    headline: str,
    offer_text: Optional[str] = None,
    cta_text: str = "Shop Now",
    logo_img: Optional[Image.Image] = None,
    size: Tuple[int,int]=(1280,720),
    color_scheme: Tuple[int,int,int]=(0,60,120)
) -> Image.Image:
    w,h = size
    banner = Image.new("RGBA", (int(w), int(h)), (0,0,0,255))
    try:
        scene = scene_img.copy().convert("RGBA")
        scene_ratio = scene.width / scene.height
        banner_ratio = w / h
        if scene_ratio > banner_ratio:
            new_h = h
            new_w = int(scene_ratio * h)
        else:
            new_w = w
            new_h = int(w / scene_ratio)
        scene = scene.resize((int(new_w), int(new_h)), Image.LANCZOS)
        banner.paste(scene, ((w-new_w)//2,(h-new_h)//2))

        # Overlay gradient
        r,g,b = color_scheme
        gradient = Image.new("L",(1,int(h)), color=0xFF)
        for y_ in range(int(h)):
            gradient.putpixel((0,y_), int(255*(y_/h)**1.5))
        alpha_gradient = gradient.resize((int(w), int(h)))
        overlay = Image.new("RGBA",(int(w), int(h)),(r,g,b,60))
        overlay.putalpha(alpha_gradient)
        banner = Image.alpha_composite(banner, overlay)

        draw = ImageDraw.Draw(banner)
        try:
            title_font = ImageFont.truetype(random.choice(FONTS), int(w*0.05))
            offer_font = ImageFont.truetype(random.choice(FONTS), int(w*0.025))
            cta_font = ImageFont.truetype(random.choice(FONTS), int(w*0.02))
        except:
            title_font = ImageFont.load_default()
            offer_font = ImageFont.load_default()
            cta_font = ImageFont.load_default()

        padding_x = int(w*0.07)
        text_y = int(h*0.55)
        max_text_width = int(w*0.85)

        # Draw headline
        text_end_y = draw_text_multiline(draw, headline, title_font, max_text_width, padding_x, text_y, line_spacing=8)

        # Gap between headline and offer
        line_gap = int(h*0.02)
        if offer_text:
            text_end_y = draw_text_multiline(draw, offer_text, offer_font, max_text_width, padding_x, text_end_y + line_gap, line_spacing=4)

        # Gap before CTA button
        cta_gap = int(h*0.03)
        cta_w = int(draw.textlength(cta_text,font=cta_font)+60)
        cta_h = int(h*0.07)
        cta_x, cta_y = padding_x, text_end_y + cta_gap
        button = Image.new("RGBA",(cta_w,cta_h),(255,255,255,255))
        mask = Image.new("L",(cta_w,cta_h),0)
        draw_mask = ImageDraw.Draw(mask)
        draw_mask.rounded_rectangle([(0,0),(cta_w,cta_h)], radius=cta_h//2, fill=255)
        banner.paste(button,(cta_x,cta_y),mask)
        text_w = draw.textlength(cta_text,font=cta_font)
        draw.text((cta_x+int((cta_w-text_w)/2), cta_y+int(cta_h/4)), cta_text, fill=color_scheme+(255,), font=cta_font)

        # Logo
        if logo_img:
            logo = logo_img.copy().convert("RGBA")
            logo.thumbnail((int(w*0.18), int(h*0.1)))
            lx,ly = w-logo.width-padding_x, padding_x
            banner.paste(logo,(lx,ly),logo)
    except Exception as e:
        st.error(f"Error creating banner: {str(e)}")
        return None
    return banner.convert("RGB")

# -------------------- Streamlit App --------------------
def main():
    st.set_page_config(page_title="Persona Ad Gen", layout="wide")
    st.title("🎨 Persona Ad Gen — Fully Autonomous Banner Generator")

    # Sidebar
    with st.sidebar:
        st.header("⚙ Configuration")
        banner_size = st.selectbox("Banner Size", list(BANNER_SIZES.keys()))
        color_scheme = st.selectbox("Color Scheme", list(COLOR_SCHEMES.keys()))

    description = st.text_area("Enter Ad Description", "Busy professionals who want to stay fit and healthy.")
    logo_file = st.file_uploader("Upload Logo (optional)", type=["png","jpg","jpeg"])

    if st.button("Generate Banners"):
        w,h = BANNER_SIZES[banner_size]
        logo_img = pil_from_bytes(logo_file.read()) if logo_file else None

        st.info("Generating AI-style headlines + offers...")
        headlines = generate_gemini_headlines(description)
        offers = random.choices(OFFER_LINES, k=len(headlines))
        ctas = random.choices(CTA_OPTIONS, k=len(headlines))

        progress_bar = st.progress(0)
        for i, (headline, offer, cta) in enumerate(zip(headlines, offers, ctas)):
            base_img = fetch_unsplash_image(description, w, h)
            if not base_img:
                base_img = Image.new("RGBA", (w,h), (50,50,50,255))

            banner = compose_banner(base_img, headline, offer, cta, logo_img,
                                    BANNER_SIZES[banner_size], COLOR_SCHEMES[color_scheme])
            if banner:
                st.image(banner, caption=f"Banner {i+1}")
                st.download_button(f"Download Banner {i+1}", data=image_to_bytes(banner),
                                   file_name=f"banner_{i+1}.png", mime="image/png")
            progress_bar.progress((i+1)/len(headlines))
            time.sleep(0.3)

        st.success(f"🎉 {len(headlines)} banners generated successfully!")

if __name__ == "__main__":
    main()
