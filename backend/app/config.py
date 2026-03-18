from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Paths
    data_dir: Path = Path("data")

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llava-phi3"
    ollama_timeout: int = 120

    # Lyrics analysis
    lyrics_enabled: bool = True
    whisper_model_size: str = "medium"

    # Upload limits
    max_images: int = 100
    max_image_size_mb: int = 20
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

    # CLIP matching
    clip_model: str = "ViT-B/32"
    clip_image_weight: float = 0.7
    clip_text_weight: float = 0.3

    # Processing
    max_workers: int = 2
    caption_parallel: int = 3

    model_config = {"env_prefix": "FLOWPIC_"}


settings = Settings()

# Ensure data directory exists
settings.data_dir.mkdir(parents=True, exist_ok=True)
