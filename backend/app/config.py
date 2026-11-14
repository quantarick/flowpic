from pathlib import Path
from typing import Optional

from pydantic import SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Paths
    data_dir: Path = Path("data")

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # LLM providers: "gemini" (hosted) or "ollama" (local).
    # llm_provider serves every role; a per-role value overrides it.
    llm_provider: str = "gemini"
    # Provider used when the chosen one cannot answer: no key, spent quota, or
    # a failed request. Empty turns the fallback off.
    llm_fallback: str = "ollama"
    vision_provider: Optional[str] = None       # image captioning, crop review
    text_provider: Optional[str] = None         # lyric emotion
    copywriting_provider: Optional[str] = None  # post copy, XHS style analysis

    # Gemini. The key is a SecretStr: it prints as ********** in logs, reprs
    # and tracebacks, and is only unwrapped to build the request header.
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-4-hydrogen"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_timeout: int = 120
    gemini_thinking_level: str = "low"  # empty string leaves the model's default

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llava-phi3"
    ollama_timeout: int = 300

    # Lyrics analysis
    lyrics_enabled: bool = True
    whisper_model_size: str = "medium"

    # Upload limits
    max_images: int = 100
    max_image_size_mb: int = 50
    max_music_size_mb: int = 100
    max_music_duration_sec: int = 600  # 10 minutes
    allowed_image_types: list[str] = ["image/jpeg", "image/png", "image/webp"]
    allowed_audio_types: list[str] = [
        "audio/mpeg", "audio/wav", "audio/flac", "audio/mp4",
        "audio/x-wav", "audio/x-flac", "audio/m4a", "audio/x-m4a",
        "audio/aac", "audio/mp4a-latm", "video/mp4",
        "application/octet-stream",
    ]

    # Video defaults
    default_fps: int = 30
    default_quality: str = "720p"
    default_aspect_ratio: str = "16:9"

    # XHS publishing
    xhs_cookie_file: str = "xhs_cookies.json"
    xhs_headless: bool = True
    xhs_style_profile_file: str = "xhs_style_profile.json"

    # CLIP matching
    clip_model: str = "ViT-B/32"
    clip_image_weight: float = 0.7
    clip_text_weight: float = 0.3

    # Processing
    max_workers: int = 2
    # Photos captioned at once. 0 picks for the provider: 1 for a local model,
    # 4 for a hosted one.
    caption_parallel: int = 0
    # Longest caption reply, in tokens, for providers that honour a cap. A
    # well-formed reply is about 170 to 300; this stops a model that ignores
    # the format and keeps writing. 0 means no cap.
    caption_max_tokens: int = 450

    # extra="ignore": a FLOWPIC_ variable with no field here (a retired
    # setting left in .env) is skipped. The default would reject it with an
    # error that prints the variable's value.
    model_config = {
        "env_prefix": "FLOWPIC_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()

# Ensure data directory exists
settings.data_dir.mkdir(parents=True, exist_ok=True)
