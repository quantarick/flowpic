"""Xiaohongshu copywriting generator using Claude API."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import anthropic

from app.config import settings
from app.models import CopywritingResult

logger = logging.getLogger(__name__)


def _load_captions(project_dir: Path) -> list[dict]:
    captions_dir = project_dir / "captions"
    if not captions_dir.exists():
        return []
    results = []
    for f in sorted(captions_dir.glob("*.json")):
        try:
            results.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    return results


def _load_json(path: Path) -> list | dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _build_prompt(captions: list[dict], emotions: list | None, locations: list | None) -> str:
    parts: list[str] = []

    # Image descriptions
    if captions:
        descs = []
        for c in captions[:30]:  # cap to avoid huge prompts
            line = c.get("caption", "")
            place = c.get("place_name")
            if place:
                line += f" (location: {place})"
            descs.append(line)
        parts.append("Image descriptions:\n" + "\n".join(f"- {d}" for d in descs))

    # Mood/emotion
    if emotions:
        moods = []
        for e in emotions[:20]:
            md = e.get("mood_description") or e.get("visual_mood_description", "")
            if md:
                moods.append(md)
        if moods:
            parts.append("Music mood keywords:\n" + ", ".join(moods))

    # Locations
    if locations:
        places = [loc.get("place_name", "") for loc in locations if loc.get("place_name")]
        if places:
            parts.append("Locations visited: " + ", ".join(places))

    return "\n\n".join(parts)


SYSTEM_PROMPT = """\
你是一位专业的小红书文案写手。根据用户提供的照片描述、旅行地点和音乐氛围信息，\
生成一篇适合小红书发布的文案。默认使用中文，但如果用户要求其他语言或风格，请按用户指示执行。

要求：
1. 标题（title）：15字以内，吸引眼球，可用emoji点缀
2. 正文（description）：100-200字，口语化、有感染力，分段落，适当使用emoji
3. 话题标签（hashtags）：5-8个热门相关标签，带#号
4. 封面推荐（cover_index）：从图片列表中选择最适合做封面的图片序号（从0开始）

用户可能会在 "User instructions" 中附加额外要求，**必须优先遵循**：
- 如果用户提供了一段已有文案并要求"续写"，则将用户文案原样放入 description 开头，接着续写至完整。
- 如果用户指定了语气、主题、语言等，按用户要求生成。
- 如果用户没有额外要求，按默认风格生成。

请严格按以下JSON格式输出，不要输出其他内容：
{"title": "...", "description": "...", "hashtags": ["#tag1", "#tag2", ...], "cover_index": 0}\
"""


def _load_style_profile() -> str | None:
    """Load cached XHS style profile and format as prompt section."""
    path = settings.data_dir / settings.xhs_style_profile_file
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("error"):
            return None
        parts = []
        if data.get("tone"):
            parts.append(f"语气风格：{data['tone']}")
        if data.get("emoji_style"):
            parts.append(f"Emoji习惯：{data['emoji_style']}")
        if data.get("sentence_structure"):
            parts.append(f"句式特点：{data['sentence_structure']}")
        if data.get("hashtag_strategy"):
            parts.append(f"标签策略：{data['hashtag_strategy']}")
        if data.get("title_pattern"):
            parts.append(f"标题风格：{data['title_pattern']}")
        if data.get("sample_phrases"):
            parts.append(f"典型短语：{'、'.join(data['sample_phrases'])}")
        if data.get("overall_summary"):
            parts.append(f"总结：{data['overall_summary']}")
        return "\n".join(parts) if parts else None
    except Exception:
        return None


def generate_copywriting(project_dir: Path, hint: str = "") -> CopywritingResult:
    """Call Claude API to generate Xiaohongshu copywriting."""
    if not settings.anthropic_api_key:
        raise ValueError("FLOWPIC_ANTHROPIC_API_KEY not configured")

    captions = _load_captions(project_dir)
    if not captions:
        raise ValueError("No image captions found. Run crop preview first.")

    cache_dir = project_dir / "cache"
    emotions = _load_json(cache_dir / "segment_emotions.json")
    locations = _load_json(cache_dir / "location_groups.json")

    user_content = _build_prompt(captions, emotions, locations)

    system = SYSTEM_PROMPT

    # Inject style profile between base prompt and user hint
    style_text = _load_style_profile()
    if style_text:
        system += f"\n\n【参考本账号的文案风格——优先模仿】\n{style_text}"

    if hint.strip():
        system += f"\n\n【用户额外要求——必须遵循】\n{hint.strip()}"

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    logger.info("Calling Claude API for copywriting (model=%s)", settings.copywriting_model)
    logger.info("System prompt:\n%s", system)
    logger.info("User content:\n%s", user_content)
    response = client.messages.create(
        model=settings.copywriting_model,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )

    text = response.content[0].text.strip()
    logger.info("Claude raw response:\n%s", text[:500])

    # Parse JSON from response — handle markdown fences and minor issues
    import re

    # Extract content between markdown code fences, or use raw text
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
    cleaned = fence_match.group(1).strip() if fence_match else text.strip()

    def _try_parse(s: str) -> dict | None:
        try:
            return json.loads(s, strict=False)
        except json.JSONDecodeError:
            return None

    def _fix_unescaped_quotes(s: str) -> str:
        """Fix unescaped double quotes inside JSON string values."""
        # Match content between key-value quote pairs:
        # "key": "value with "unescaped" quotes"
        # Strategy: find each "key": "..." pattern and escape inner quotes
        def fix_value(m):
            prefix = m.group(1)  # "key": "
            content = m.group(2)  # value content
            suffix = m.group(3)  # closing pattern
            fixed = content.replace('"', '\u201c').replace('"', '\u201d')
            return prefix + fixed + suffix
        # Match "key": "value" where value may contain unescaped quotes
        return re.sub(
            r'("(?:title|description)":\s*")(.*?)("\s*[,}])',
            fix_value,
            s,
            flags=re.DOTALL,
        )

    data = _try_parse(cleaned) or _try_parse(text)

    if data is None:
        # Try fixing unescaped quotes
        data = _try_parse(_fix_unescaped_quotes(cleaned))

    if data is None:
        # Extract the outermost {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            data = _try_parse(m.group()) or _try_parse(_fix_unescaped_quotes(m.group()))

    if data is None:
        raise ValueError(f"Claude returned non-JSON response: {text[:300]}")

    result = CopywritingResult(
        title=data.get("title", ""),
        description=data.get("description", ""),
        hashtags=data.get("hashtags", []),
        cover_index=data.get("cover_index", 0),
    )

    # Cache result
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "copywriting.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )

    logger.info("Copywriting generated and cached")
    return result


def get_cached_copywriting(project_dir: Path) -> CopywritingResult | None:
    """Return cached copywriting result if it exists."""
    path = project_dir / "cache" / "copywriting.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return CopywritingResult(**data)
    except Exception:
        return None
