"""Scrape user's XHS posts and analyze writing style via Claude."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from app.config import settings
from app.models import ScrapedPost, XhsStyleProfile

logger = logging.getLogger(__name__)

MAX_POSTS = 5


def _style_profile_path() -> Path:
    return settings.data_dir / settings.xhs_style_profile_file


def _cookie_path() -> Path:
    return settings.data_dir / settings.xhs_cookie_file


def _load_cookies() -> tuple[str, list[dict] | None]:
    """Load cookies from stored file. Returns (cookie_str, pw_cookies)."""
    path = _cookie_path()
    if not path.exists():
        raise ValueError("No XHS cookies configured. Connect your account first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    cookie_str = data.get("cookie", "")
    pw_cookies = data.get("playwright_cookies")
    if pw_cookies:
        for c in pw_cookies:
            if c.get("sameSite") == "None" and not c.get("secure"):
                c["sameSite"] = "Lax"
    return cookie_str, pw_cookies


def _extract_user_id(cookie_str: str, pw_cookies: list[dict] | None) -> str | None:
    """Extract user_id from cookies."""
    # Check pw_cookies first for x-user-id cookie variants
    if pw_cookies:
        for c in pw_cookies:
            name = c.get("name", "")
            if "user-id" in name.lower() or name == "customerClientId":
                return c["value"]
    # Parse cookie string
    for part in cookie_str.split(";"):
        part = part.strip()
        if "=" in part:
            key, _, value = part.partition("=")
            key = key.strip()
            if "user-id" in key.lower() or key == "customerClientId":
                return value.strip()
    return None


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
            "domain": ".xiaohongshu.com",
            "path": "/",
            "secure": True,
            "sameSite": "None",
        })
    return cookies


async def _scrape_posts(cookie_str: str, pw_cookies: list[dict] | None) -> list[ScrapedPost]:
    """Use Playwright to scrape the user's latest posts."""
    from playwright.async_api import async_playwright

    user_id = _extract_user_id(cookie_str, pw_cookies)
    if not user_id:
        raise ValueError(
            "Could not find user ID in cookies. "
            "Make sure your cookie export includes the user ID cookie."
        )

    if not pw_cookies:
        pw_cookies = _parse_cookies_for_playwright(cookie_str)

    profile_url = f"https://www.xiaohongshu.com/user/profile/{user_id}"
    logger.info("Scraping XHS profile: %s", profile_url)

    posts: list[ScrapedPost] = []

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
            page = await context.new_page()

            # Navigate to profile
            await page.goto(profile_url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(3)

            # Check for login redirect
            if "login" in page.url or "accounts" in page.url:
                raise ValueError("Cookie expired — redirected to login page.")

            # Find note links on profile page
            note_links = await page.evaluate("""() => {
                const links = [];
                const anchors = document.querySelectorAll('a[href*="/explore/"], a[href*="/discovery/item/"]');
                for (const a of anchors) {
                    const href = a.getAttribute('href');
                    if (href && !links.includes(href)) {
                        links.push(href);
                    }
                    if (links.length >= 5) break;
                }
                // Also try section.note-item links
                if (links.length === 0) {
                    const items = document.querySelectorAll('section.note-item a, div.note-item a, [class*="note"] a');
                    for (const a of items) {
                        const href = a.getAttribute('href');
                        if (href && href.includes('/') && !links.includes(href)) {
                            links.push(href);
                        }
                        if (links.length >= 5) break;
                    }
                }
                return links;
            }""")

            logger.info("Found %d note links on profile", len(note_links))

            # Visit each note and extract content
            for link in note_links[:MAX_POSTS]:
                try:
                    # Normalize URL
                    if link.startswith("/"):
                        url = f"https://www.xiaohongshu.com{link}"
                    elif not link.startswith("http"):
                        url = f"https://www.xiaohongshu.com/{link}"
                    else:
                        url = link

                    await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                    await asyncio.sleep(2)

                    note_data = await page.evaluate("""() => {
                        const title = document.querySelector('#detail-title, .title, [class*="title"]');
                        const desc = document.querySelector('#detail-desc, .desc, .content, [class*="desc"]');
                        const tagEls = document.querySelectorAll('.tag, a[href*="search_result"], [class*="hashtag"], [class*="tag"]');
                        const tags = [];
                        for (const t of tagEls) {
                            const text = t.textContent.trim();
                            if (text.startsWith('#') || text.startsWith('# ')) {
                                tags.push(text);
                            }
                        }
                        return {
                            title: title ? title.textContent.trim() : '',
                            description: desc ? desc.textContent.trim() : '',
                            hashtags: tags,
                        };
                    }""")

                    if note_data["title"] or note_data["description"]:
                        posts.append(ScrapedPost(
                            title=note_data["title"],
                            description=note_data["description"],
                            hashtags=note_data["hashtags"],
                            note_url=url,
                        ))
                        logger.info("Scraped note: %s", note_data["title"][:30])

                except Exception as e:
                    logger.warning("Failed to scrape note %s: %s", link, e)
                    continue

        finally:
            await browser.close()

    return posts


def _analyze_style(posts: list[ScrapedPost]) -> XhsStyleProfile:
    """Send scraped posts to Claude for style analysis."""
    if not settings.anthropic_api_key:
        raise ValueError("FLOWPIC_ANTHROPIC_API_KEY not configured")

    posts_text = ""
    for i, post in enumerate(posts, 1):
        posts_text += f"\n--- 帖子 {i} ---\n"
        posts_text += f"标题：{post.title}\n"
        posts_text += f"正文：{post.description}\n"
        if post.hashtags:
            posts_text += f"标签：{' '.join(post.hashtags)}\n"

    system_prompt = """\
你是文案风格分析专家。分析以下小红书帖子，总结这位博主的写作风格特征。

请严格按以下JSON格式输出，不要输出其他内容：
{
  "tone": "语气风格描述（如：轻松活泼/文艺清新/专业理性/热情奔放）",
  "emoji_style": "emoji使用习惯描述（如：大量使用、点缀使用、几乎不用）",
  "sentence_structure": "句式特点（如：短句为主、长短交错、排比句多）",
  "hashtag_strategy": "标签策略（如：热门标签+小众标签组合、纯地点标签、情绪标签为主）",
  "title_pattern": "标题风格（如：疑问句式、感叹句式、数字+关键词、emoji开头）",
  "sample_phrases": ["从原文提取3-5个最能代表此博主风格的短语或句式"],
  "overall_summary": "50字以内的整体风格总结，用于指导后续文案生成"
}\
"""

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    logger.info("Analyzing style with Claude (%d posts)", len(posts))

    response = client.messages.create(
        model=settings.copywriting_model,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": posts_text}],
    )

    text = response.content[0].text.strip()
    logger.info("Style analysis raw response: %s", text[:300])

    # Parse JSON
    import re
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
    cleaned = fence_match.group(1).strip() if fence_match else text.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try extracting {...}
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            data = json.loads(m.group())
        else:
            raise ValueError(f"Claude returned non-JSON response: {text[:300]}")

    return XhsStyleProfile(
        tone=data.get("tone", ""),
        emoji_style=data.get("emoji_style", ""),
        sentence_structure=data.get("sentence_structure", ""),
        hashtag_strategy=data.get("hashtag_strategy", ""),
        title_pattern=data.get("title_pattern", ""),
        sample_phrases=data.get("sample_phrases", []),
        overall_summary=data.get("overall_summary", ""),
        scraped_posts=posts,
        scraped_at=datetime.now(timezone.utc).isoformat(),
    )


# ─── Public API ───────────────────────────────────────────────

def get_cached_style_profile() -> XhsStyleProfile | None:
    """Return cached style profile if it exists."""
    path = _style_profile_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return XhsStyleProfile(**data)
    except Exception:
        return None


def clear_style_profile():
    """Delete cached style profile."""
    path = _style_profile_path()
    if path.exists():
        path.unlink()
        logger.info("Cleared style profile cache")


def scrape_user_style(force: bool = False) -> XhsStyleProfile:
    """Scrape user's latest posts and analyze writing style.

    Returns cached profile unless force=True.
    """
    if not force:
        cached = get_cached_style_profile()
        if cached:
            logger.info("Returning cached style profile (scraped_at=%s)", cached.scraped_at)
            return cached

    # Load cookies and scrape
    cookie_str, pw_cookies = _load_cookies()

    # Run Playwright scraping
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        def _run():
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            return asyncio.run(_scrape_posts(cookie_str, pw_cookies))
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            posts = pool.submit(_run).result(timeout=120)
    else:
        posts = asyncio.run(_scrape_posts(cookie_str, pw_cookies))

    if not posts:
        return XhsStyleProfile(
            tone="", emoji_style="", sentence_structure="",
            hashtag_strategy="", title_pattern="", overall_summary="",
            error="No posts found on profile. Make sure you have published posts.",
        )

    # Analyze with Claude
    profile = _analyze_style(posts)

    # Cache to disk
    path = _style_profile_path()
    path.write_text(
        profile.model_dump_json(indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Style profile cached to %s", path)

    return profile
