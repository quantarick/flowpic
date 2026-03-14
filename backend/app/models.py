from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --- Enums ---

class AspectRatio(str, Enum):
    RATIO_16_9 = "16:9"
    RATIO_21_9 = "21:9"
    RATIO_9_16 = "9:16"
    RATIO_1_1 = "1:1"
    RATIO_4_3 = "4:3"


class Quality(str, Enum):
    SD = "720p"
    HD = "1080p"
    QHD = "2k"
    UHD = "4k"


class TaskStatus(str, Enum):
    PENDING = "pending"
    ANALYZING_AUDIO = "analyzing_audio"
    ANALYZING_LYRICS = "analyzing_lyrics"
    CLASSIFYING_EMOTION = "classifying_emotion"
    CAPTIONING_IMAGES = "captioning_images"
    MATCHING = "matching"
    RENDERING = "rendering"
    ENCODING = "encoding"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


# --- Request / Response models ---

class ProjectConfig(BaseModel):
    aspect_ratio: AspectRatio = AspectRatio.RATIO_16_9
    quality: Quality = Quality.SD
    fps: int = Field(default=30, ge=15, le=60)
    vision_model: Optional[str] = None  # Ollama model override; None = use server default


class ProjectCreateResponse(BaseModel):
    project_id: str


class ProjectInfo(BaseModel):
    project_id: str
    images: list[str] = []
    music: Optional[str] = None
    config: ProjectConfig = Field(default_factory=ProjectConfig)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GenerateResponse(BaseModel):
    task_id: str


class UploadResponse(BaseModel):
    filenames: list[str]
    count: int


class MusicUploadResponse(BaseModel):
    filename: str
    duration_seconds: Optional[float] = None


# --- Progress / WebSocket ---

class ProgressMessage(BaseModel):
    status: TaskStatus
    progress: float = Field(ge=0, le=100)
    current_step: str = ""
    detail: str = ""
    eta_seconds: Optional[float] = None


# --- Internal pipeline models ---

class AudioSegment(BaseModel):
    start: float
    end: float
    beat_count: int
    rms_energy: float
    spectral_centroid: float
    chroma: list[float]


class AudioFeatures(BaseModel):
    beat_times: list[float]
    onset_times: list[float]
    tempo: float
    duration: float
    segments: list[AudioSegment]


class EmotionResult(BaseModel):
    valence: float
    arousal: float
    mood_description: str


class SegmentEmotion(BaseModel):
    segment_index: int
    start: float
    end: float
    valence: float
    arousal: float
    mood_description: str


class FaceRegion(BaseModel):
    x: int
    y: int
    w: int
    h: int


class ImageCaption(BaseModel):
    filename: str
    caption: str
    face_regions: list[FaceRegion] = []
    has_person: bool = False
    focus_x: float = 0.5
    focus_y: float = 0.5
    fit_mode: str = "crop"  # "crop" or "full" (blur fill)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    place_name: Optional[str] = None


class MatchResult(BaseModel):
    segment_index: int
    image_filename: str
    similarity_score: float


class TranscribedWord(BaseModel):
    word: str
    start: float
    end: float
    probability: float


class LyricsResult(BaseModel):
    text: str
    words: list[TranscribedWord]
    language: str
    has_vocals: bool


class SegmentLyrics(BaseModel):
    segment_index: int
    start: float
    end: float
    text: str
    word_count: int


class LyricEmotion(BaseModel):
    segment_index: int
    theme: str
    mood_keywords: list[str]
    mood_description: str


class TaskRecord(BaseModel):
    """Persisted task record for history."""
    task_id: str
    project_id: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    output_path: Optional[str] = None
    image_count: int = 0
    duration_seconds: Optional[float] = None
    config: Optional[ProjectConfig] = None
    error_message: Optional[str] = None


class LocationGroup(BaseModel):
    place_name: str
    start_clip_index: int
    end_clip_index: int


class KenBurnsParams(BaseModel):
    zoom_start: float
    zoom_end: float
    pan_x_start: float
    pan_x_end: float
    pan_y_start: float
    pan_y_end: float
    face_center: Optional[tuple[float, float]] = None
