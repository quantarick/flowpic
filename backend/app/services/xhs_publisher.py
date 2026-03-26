"""Xiaohongshu publishing service using Playwright browser automation."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from pathlib import Path

from app.config import settings
from app.models import XhsCookieStatus, XhsPublishResult

logger = logging.getLogger(__name__)

XHS_CREATOR_URL = "https://creator.xiaohongshu.com/publish/publish?source=web&type=normal"
XHS_DOMAIN = ".xiaohongshu.com"

# DOM selectors for the XHS creator page
SELECTORS = {
    "file_input": 'input[type="file"]',
    "image_preview": ".img-container",
    "title_input": 'input.d-text[placeholder]',
    "content_editor": "div.ProseMirror[contenteditable]",
    "publish_button": "button.bg-red",
}

# Max title length enforced by XHS
MAX_TITLE_LEN = 20
MAX_IMAGES = 18


def _cookie_path() -> Path:
    return settings.data_dir / settings.xhs_cookie_file


def _is_json_cookie_export(text: str) -> bool:
    """Detect JSON cookie export format (browser extension array of objects)."""
    stripped = text.strip()
    return stripped.startswith("[") and stripped.endswith("]")


def _parse_json_cookie_export(text: str) -> list[dict]:
    """Parse JSON cookie export (e.g. EditThisCookie) into Playwright cookie dicts."""
    raw = json.loads(text)
    cookies = []
    for c in raw:
        name = c.get("name", "")
        value = c.get("value", "")
        if not name:
            continue
        secure = c.get("secure", False)
        # Map sameSite: extension uses "unspecified"/"lax"/"strict"/"no_restriction"
        same_site = c.get("sameSite", "unspecified")
        if same_site in ("unspecified", "no_restriction"):
            # Playwright requires secure=true for sameSite="None"
            same_site = "None" if secure else "Lax"
        elif same_site == "lax":
            same_site = "Lax"
        elif same_site == "strict":
            same_site = "Strict"
        cookies.append({
            "name": name,
            "value": value,
            "domain": c.get("domain", XHS_DOMAIN),
            "path": c.get("path", "/"),
            "secure": secure,
            "sameSite": same_site,
        })
    return cookies


def _parse_cookie_string(cookie_str: str) -> dict:
    """Parse a raw cookie header string into a dict."""
    cookies = {}
    for part in cookie_str.split(";"):
        part = part.strip()
        if "=" in part:
            key, _, value = part.partition("=")
            cookies[key.strip()] = value.strip()
    return cookies


def _pw_cookies_to_dict(pw_cookies: list[dict]) -> dict:
    """Convert Playwright cookie list to a simple name→value dict."""
    return {c["name"]: c["value"] for c in pw_cookies}


def _parse_cookies_for_playwright(cookie_str: str) -> list[dict]:
    """Convert raw cookie header string to Playwright cookie format."""
    cookies = []
    for part in cookie_str.split(";"):
        part = part.strip()
        if "=" not in part:
            continue
        key, _, value = part.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        cookies.append({
            "name": key,
            "value": value,
            "domain": XHS_DOMAIN,
            "path": "/",
            "secure": True,
            "sameSite": "None",
        })
    return cookies


async def _human_delay(min_ms: int = 500, max_ms: int = 1500):
    """Small random delay to appear more human."""
    await asyncio.sleep(random.randint(min_ms, max_ms) / 1000)


async def _type_like_human(element, text: str):
    """Type text with random per-keystroke delay."""
    await element.type(text, delay=random.randint(30, 80))


async def _publish_via_browser(
    cookie_str: str,
    image_paths: list[str],
    title: str,
    description: str,
    hashtags: list[str],
    pw_cookies: list[dict] | None = None,
) -> XhsPublishResult:
    """Automate XHS publishing via Playwright."""
    from playwright.async_api import async_playwright

    if not pw_cookies:
        pw_cookies = _parse_cookies_for_playwright(cookie_str)
    if not pw_cookies:
        return XhsPublishResult(error="No valid cookies to inject")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=settings.xhs_headless)
        try:
            context = await browser.new_context(
                viewport={"width": 1280, "height": 900},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
            )
            await context.add_cookies(pw_cookies)
            # Verify which cookies were actually set
            actual = await context.cookies()
            logger.info(
                "Injected %d cookies, browser has %d: %s",
                len(pw_cookies), len(actual),
                [c["name"] for c in actual],
            )
            page = await context.new_page()

            # Navigate to creator page
            logger.info("Navigating to XHS creator page")
            await page.goto(XHS_CREATOR_URL, wait_until="domcontentloaded", timeout=30000)
            # Wait for page to stabilize (networkidle hangs on XHS)
            await _human_delay(3000, 5000)

            # Check if redirected to login
            current_url = page.url
            if "login" in current_url or "accounts" in current_url:
                debug_dir = settings.data_dir / "xhs_debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                await page.screenshot(path=str(debug_dir / "00_login_redirect.png"))
                logger.error("Redirected to login: %s", current_url)
                return XhsPublishResult(
                    error="Cookie expired or invalid. Please reconnect your XHS account."
                )

            # Debug screenshot
            debug_dir = settings.data_dir / "xhs_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            await page.screenshot(path=str(debug_dir / "01_page_loaded.png"))
            logger.info("Page loaded at: %s", current_url)

            # Switch to "上传图文" (image+text) tab — page defaults to video tab
            # Use JS click because the tab can be outside viewport with sidebar open
            await page.evaluate("""
                for (const el of document.querySelectorAll('span')) {
                    if (el.textContent.includes('上传图文')) { el.click(); break; }
                }
            """)
            logger.info("Switched to image upload tab")
            await _human_delay(1500, 2500)

            # Upload images via the image file input
            logger.info("Uploading %d images", len(image_paths))
            file_input = page.locator('input[type="file"][accept*="jpg"]')
            await file_input.first.set_input_files(image_paths)
            await _human_delay(1000, 2000)

            # Wait for image previews to appear
            try:
                await page.locator(SELECTORS["image_preview"]).first.wait_for(
                    state="visible", timeout=60000
                )
                logger.info("Image previews visible")
            except Exception:
                await page.screenshot(path=str(debug_dir / "02_upload_timeout.png"))
                return XhsPublishResult(
                    error="Image upload timed out. Check debug screenshots."
                )

            await _human_delay()
            await page.screenshot(path=str(debug_dir / "02_images_uploaded.png"))

            # Fill title (truncate if needed)
            if title:
                truncated_title = title[:MAX_TITLE_LEN]
                logger.info("Setting title: %s", truncated_title)
                title_input = page.locator(SELECTORS["title_input"]).first
                await title_input.click()
                await _human_delay(200, 500)
                await _type_like_human(title_input, truncated_title)
                await _human_delay()

            # Fill description in content editor
            if description or hashtags:
                logger.info("Setting description")
                editor = page.locator(SELECTORS["content_editor"]).first
                await editor.click()
                await _human_delay(200, 500)

                # Type description
                if description:
                    await _type_like_human(editor, description)
                    await _human_delay()

                # Add hashtags
                if hashtags:
                    # Add newline before hashtags if there's description
                    if description:
                        await editor.press("Enter")
                        await _human_delay(200, 500)

                    for tag in hashtags:
                        tag_text = tag if tag.startswith("#") else f"#{tag}"
                        await _type_like_human(editor, tag_text)
                        await _human_delay(300, 800)
                        # Wait for tag suggestion popup, then press Enter to confirm
                        try:
                            await page.wait_for_selector(
                                ".publish-hash-tag", timeout=3000
                            )
                            await page.keyboard.press("Enter")
                        except Exception:
                            # No suggestion popup, just add a space
                            await editor.press("Space")
                        await _human_delay(300, 600)

            await page.screenshot(path=str(debug_dir / "03_form_filled.png"))
            await _human_delay(500, 1000)

            # Click publish button
            logger.info("Clicking publish button")
            publish_btn = page.locator(SELECTORS["publish_button"]).first
            try:
                await publish_btn.wait_for(state="visible", timeout=5000)
            except Exception:
                # Try alternative selector
                publish_btn = page.locator("button:has-text('发布')").first
                try:
                    await publish_btn.wait_for(state="visible", timeout=5000)
                except Exception:
                    await page.screenshot(
                        path=str(debug_dir / "04_no_publish_btn.png")
                    )
                    return XhsPublishResult(
                        error="Publish button not found. Check debug screenshots."
                    )

            await publish_btn.click()
            logger.info("Publish button clicked, waiting for result")

            # Wait for navigation or success indication
            try:
                await page.wait_for_url(
                    "**/publish/success**", timeout=30000
                )
                logger.info("Publish success detected via URL")
            except Exception:
                # May not redirect — check for error modals or other indicators
                await _human_delay(2000, 3000)
                await page.screenshot(path=str(debug_dir / "04_after_publish.png"))

                # Check for common error patterns
                error_el = page.locator(".error-tip, .toast-error, .d-toast")
                if await error_el.count() > 0:
                    error_text = await error_el.first.inner_text()
                    logger.error("Publish error: %s", error_text)
                    return XhsPublishResult(error=f"XHS error: {error_text}")

                # Check if we got CAPTCHA / verification
                if "verify" in page.url or "captcha" in page.url.lower():
                    return XhsPublishResult(
                        error="CAPTCHA/verification required. Please publish manually."
                    )

            await page.screenshot(path=str(debug_dir / "05_final.png"))
            final_url = page.url

            # Try to extract note ID from URL
            note_id = ""
            if "success" in final_url:
                # URL might contain the note ID
                parts = final_url.split("/")
                for i, part in enumerate(parts):
                    if part == "success" and i + 1 < len(parts):
                        note_id = parts[i + 1].split("?")[0]
                        break

            post_url = (
                f"https://www.xiaohongshu.com/explore/{note_id}" if note_id else None
            )

            logger.info("Published successfully. note_id=%s url=%s", note_id, post_url)
            return XhsPublishResult(
                success=True,
                post_url=post_url,
                note_id=note_id or None,
            )

        except Exception as e:
            logger.error("Playwright publish failed: %s", e, exc_info=True)
            # Try to capture screenshot on failure
            try:
                await page.screenshot(
                    path=str(debug_dir / "error_final.png")
                )
            except Exception:
                pass
            return XhsPublishResult(error=f"Browser automation failed: {e}")
        finally:
            await browser.close()


# ─── Public API (unchanged contracts) ───────────────────────────────

def save_cookies(cookie_str: str) -> XhsCookieStatus:
    """Parse raw cookie string (header format or JSON export), save to JSON."""
    cookie_str = cookie_str.strip()
    if not cookie_str:
        return XhsCookieStatus(error="Cookie string is empty")

    if _is_json_cookie_export(cookie_str):
        try:
            pw_cookies = _parse_json_cookie_export(cookie_str)
        except (json.JSONDecodeError, TypeError) as e:
            return XhsCookieStatus(error=f"Invalid JSON cookie format: {e}")
        if not pw_cookies:
            return XhsCookieStatus(error="No cookies found in JSON export")
        parsed = _pw_cookies_to_dict(pw_cookies)
    else:
        parsed = _parse_cookie_string(cookie_str)
        pw_cookies = None

    if "a1" not in parsed or "web_session" not in parsed:
        return XhsCookieStatus(
            error="Cookie must contain 'a1' and 'web_session' fields"
        )

    # Store both the header string and structured Playwright cookies
    store: dict = {}
    if pw_cookies:
        store["cookie"] = "; ".join(f"{c['name']}={c['value']}" for c in pw_cookies)
        store["playwright_cookies"] = pw_cookies
        logger.info("Saved %d cookies from JSON export", len(pw_cookies))
    else:
        store["cookie"] = cookie_str

    path = _cookie_path()
    path.write_text(
        json.dumps(store, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return XhsCookieStatus(
        connected=True,
        username=parsed.get("web_session", "")[:8] + "...",
    )


def validate_cookies() -> XhsCookieStatus:
    """Load stored cookies and check if they're saved (lightweight check)."""
    path = _cookie_path()
    if not path.exists():
        return XhsCookieStatus(connected=False)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cookie_str = data.get("cookie", "")
        if not cookie_str:
            return XhsCookieStatus(connected=False)

        parsed = _parse_cookie_string(cookie_str)
        if "a1" not in parsed or "web_session" not in parsed:
            return XhsCookieStatus(connected=False, error="Invalid cookie format")

        return XhsCookieStatus(
            connected=True,
            username=parsed.get("web_session", "")[:8] + "...",
        )
    except Exception as e:
        return XhsCookieStatus(connected=False, error=str(e))


def clear_cookies() -> None:
    """Delete stored cookie file."""
    path = _cookie_path()
    if path.exists():
        path.unlink()


def publish_note(
    project_dir: Path,
    title: str,
    description: str,
    hashtags: list[str],
    image_filenames: list[str] | None = None,
) -> XhsPublishResult:
    """Upload images and create an image note on XHS via browser automation."""
    # Load cookies
    path = _cookie_path()
    if not path.exists():
        return XhsPublishResult(error="No XHS cookies configured. Connect your account first.")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cookie_str = data.get("cookie", "")
        pw_cookies = data.get("playwright_cookies")
        # Fix sameSite for stored cookies: Playwright rejects sameSite=None + secure=False
        if pw_cookies:
            for c in pw_cookies:
                if c.get("sameSite") == "None" and not c.get("secure"):
                    c["sameSite"] = "Lax"
        if not cookie_str and not pw_cookies:
            return XhsPublishResult(error="Stored cookie is empty.")
    except Exception as e:
        return XhsPublishResult(error=f"Failed to read cookies: {e}")

    # Collect image paths
    crops_dir = project_dir / "output" / "crops"
    image_paths: list[str] = []

    if image_filenames:
        for fname in image_filenames:
            p = crops_dir / fname
            if p.exists():
                image_paths.append(str(p))
            else:
                logger.warning("Image not found, skipping: %s", p)

    if not image_paths:
        crop_files = sorted(crops_dir.glob("*.png")) + sorted(crops_dir.glob("*.jpg"))
        if not crop_files:
            return XhsPublishResult(error="No crop images found to publish")
        image_paths = [str(f) for f in crop_files]

    if len(image_paths) > MAX_IMAGES:
        logger.warning("Truncating from %d to %d images", len(image_paths), MAX_IMAGES)
        image_paths = image_paths[:MAX_IMAGES]

    logger.info("Publishing %d images to XHS via browser", len(image_paths))

    # Run async Playwright in a sync context
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an async context (FastAPI) — run in a new thread.
        # On Windows, new threads get SelectorEventLoop which doesn't support
        # subprocesses (Playwright needs them), so we force ProactorEventLoop.
        import concurrent.futures
        import sys
        def _run_playwright():
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            return asyncio.run(
                _publish_via_browser(cookie_str, image_paths, title, description, hashtags, pw_cookies)
            )
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(_run_playwright).result(timeout=120)
        return result
    else:
        return asyncio.run(
            _publish_via_browser(cookie_str, image_paths, title, description, hashtags, pw_cookies)
        )
