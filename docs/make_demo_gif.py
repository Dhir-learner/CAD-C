"""Record docs/demo.gif: a short tour of the web app (needs the app running and Playwright + Edge).

    .venv\\Scripts\\python -m app.server --port 8765 --no-browser
    uv run --no-project --with playwright --with pillow python docs/make_demo_gif.py
"""
import asyncio
import io
from pathlib import Path

from PIL import Image
from playwright.async_api import async_playwright

URL = "http://127.0.0.1:8765"
OUT = Path(__file__).with_name("demo.gif")
WIDTH = 960  # output width in pixels


async def main():
    frames = []  # (image, duration ms)

    async def grab(page, ms, clip=None):
        png = await page.screenshot(clip=clip)
        img = Image.open(io.BytesIO(png)).convert("RGB")
        img = img.resize((WIDTH, round(img.height * WIDTH / img.width)), Image.LANCZOS)
        frames.append((img, ms))

    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="msedge")
        ctx = await browser.new_context(viewport={"width": 1440, "height": 900}, color_scheme="dark")
        await ctx.add_init_script("try{localStorage.setItem('cadc-consent-v1','yes')}catch(e){}")
        page = await ctx.new_page()

        # 1. Home: the rotating 3D lungs
        await page.goto(URL)
        await page.wait_for_timeout(1500)
        for _ in range(8):
            await grab(page, 180)
            await page.wait_for_timeout(180)

        # 2. Learn: cancer stages on the 3D model
        await page.goto(URL + "/learn")
        await page.wait_for_timeout(800)
        await page.evaluate("document.getElementById('explore').scrollIntoView()")
        await page.evaluate("window.scrollBy(0, -80)")
        await page.wait_for_timeout(1200)
        for s in range(5):
            await page.click(f"#stageBtns [data-s='{s}']")
            await page.wait_for_timeout(900)
            await grab(page, 1100)

        # 3. Analyse: sample scan, automatic detection, result with AI attention
        await page.goto(URL + "/analyze")
        await page.wait_for_timeout(600)
        await grab(page, 1200)
        await page.click("#demoBtn")
        await page.wait_for_selector("#viewer.on", timeout=120000)
        await page.evaluate("window.scrollTo(0, 280)")
        await page.wait_for_timeout(600)
        await grab(page, 1200)
        await page.click("#detectBtn")
        await page.wait_for_selector("#cands .chip", timeout=300000)
        await page.wait_for_timeout(600)
        await page.evaluate("window.scrollTo(0, 280)")
        await grab(page, 1800)
        await page.click("#cands .chip")
        await page.wait_for_selector("#result.on", timeout=120000)
        await page.wait_for_timeout(1400)
        await page.evaluate("window.scrollTo(0, 560)")
        await page.wait_for_timeout(300)
        await grab(page, 2200)
        await page.click("#viewSeg [data-v=heatmaps]")
        await page.evaluate("document.getElementById('viewSeg').scrollIntoView({block: 'center'})")
        await page.wait_for_timeout(500)
        await grab(page, 2200)

        # 4. Model page: comparison and charts
        await page.goto(URL + "/model")
        await page.wait_for_timeout(1500)
        await page.evaluate("document.querySelectorAll('.reveal').forEach(e => e.classList.add('in'))")
        await grab(page, 2000)
        await page.evaluate("window.scrollTo(0, 780)")
        await page.wait_for_timeout(600)
        await grab(page, 2500)
        await browser.close()

    images = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f, _ in frames]
    images[0].save(OUT, save_all=True, append_images=images[1:], duration=[d for _, d in frames], loop=0, optimize=True)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(images)} frames)")


if __name__ == "__main__":
    asyncio.run(main())
