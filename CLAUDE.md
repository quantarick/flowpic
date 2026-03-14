# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend
cd frontend
npm install
npm run dev          # dev server at :5173
npm run build        # production build → frontend/dist/

# Docker (all services)
docker compose up

# Tests
cd backend && python -m pytest tests/

# Ollama (must be running for image captioning)
ollama serve
ollama pull moondream
```

## Architecture

FlowPic is a music-driven slideshow generator with a Python backend (FastAPI) and React frontend.

### Pipeline (backend/app/workers/video_worker.py)

The video generation pipeline runs as a background thread and flows through these stages:

1. **AudioAnalyzer** (`services/audio_analyzer.py`) — librosa beat/onset/tempo detection, segments song at 4-beat boundaries (2-6s each), extracts RMS energy + spectral centroid + chroma per segment
2. **EmotionClassifier** (`services/emotion_classifier.py`) — Music2Emo model predicts global valence/arousal, then modulates per-segment based on energy/spectral features. Outputs natural language mood descriptions. Has fallback if Music2Emo fails.
3. **Lyrics pipeline** (optional, when `lyrics_enabled=True`):
   - `VocalSeparator` → Demucs vocal isolation
   - `LyricsTranscriber` → Whisper speech-to-text
   - `LyricEmotionAnalyzer` → per-segment lyric emotion via Ollama
   - Models load/unload sequentially to keep VRAM ≤ 2.5GB
4. **ImageCaptioner** (`services/image_captioner.py`) — Ollama moondream generates scene/mood descriptions. Also detects faces (Haar cascade) and bodies (HOG). Captions cached as JSON in project dir.
5. **Deduplication** — dHash perceptual hashing removes near-duplicates, keeps sharpest
6. **SemanticMatcher** (`services/matcher.py`) — sentence-transformers encodes mood descriptions + image captions, cosine similarity matrix, Hungarian algorithm for optimal one-to-one assignment. Merges low-energy segments when images < segments.
7. **Location clustering** — Groups GPS-tagged images by place, computes LocationGroups for title cards
8. **VideoGenerator** (`services/video_generator.py`) — Ken Burns effect (1.3x oversized source, cubic-eased zoom/pan), manual crossfade at beat boundaries, smart crop with face detection, location title cards for groups >3 clips. Assembles via moviepy + ffmpeg.

### Key data flow
```
Music → AudioFeatures → SegmentEmotion[] (mood descriptions)
Images → ImageCaption[] (scene descriptions + GPS + faces)
SegmentEmotion[] × ImageCaption[] → cosine similarity → MatchResult[]
MatchResult[] → location clustering → LocationGroup[] → VideoGenerator → MP4
```

### Backend structure
- `app/main.py` — FastAPI app with CORS, lifespan (TaskManager init), router includes
- `app/config.py` — Pydantic Settings with `FLOWPIC_` env prefix
- `app/models.py` — All Pydantic models (AudioFeatures, SegmentEmotion, ImageCaption, MatchResult, ProjectConfig, TaskStatus enum, LocationGroup, etc.)
- `app/routers/` — upload, project, generate (includes WebSocket at `/ws`), download
- `app/services/` — Each service is a class with lazy model loading; GPU models unload after use
- `app/core/ken_burns.py` — Pure math: keyframe generation, crop interpolation, cubic easing
- `app/core/transitions.py` — Frame blending, beat-snapped timing
- `app/workers/video_worker.py` — Pipeline orchestration with progress callbacks and cancellation

### Frontend structure
- React 18 + TypeScript + Vite
- `hooks/useProject.ts` — Central state (projectId, images, music, config, taskId, status)
- `hooks/useWebSocket.ts` — WebSocket connection to `/ws` for real-time progress
- `components/` — ImageUploader, MusicUploader, ConfigPanel, GenerateButton, ProgressBar, VideoPreview, TaskHistory, LangSwitch
- `i18n/` — English (`en.ts`) and Chinese (`zh.ts`) translations
- `api/client.ts` — fetch wrappers for REST endpoints

### Progress reporting
WebSocket at `/ws` pushes `ProgressMessage` JSON with status enum, progress percentage (0-100), current_step, and detail text. Client can send `{action: "cancel"}`.

### Data storage
File-based per project: `backend/data/{project_id}/` with `meta.json`, `images/`, `music.*`, `captions/*.json`, `output/*.mp4`. Task history in SQLite `backend/data/tasks.db`.

## Key patterns

- GPU models (Music2Emo, Demucs, Whisper) load/unload sequentially to share VRAM
- Image captions are cached in `{project_dir}/captions/{hash}.json` — skip on regenerate
- Uploaded files are cleaned up after video generation; caption cache kept for debugging
- Services use lazy imports inside `video_worker.py` to avoid loading all models at startup
- Config via `pydantic-settings` with `FLOWPIC_` env prefix (e.g., `FLOWPIC_OLLAMA_MODEL`)
- Aspect ratios: 16:9, 21:9, 9:16, 1:1, 4:3. Quality: 720p, 1080p, 2K, 4K
