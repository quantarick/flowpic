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

# Ollama (only needed when a provider setting is "ollama"; the default LLM is Gemini)
ollama serve
ollama pull llava-phi3
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
   - `LyricEmotionAnalyzer` → per-segment lyric emotion via the text LLM
   - Models load/unload sequentially to keep VRAM ≤ 2.5GB
4. **ImageCaptioner** (`services/image_captioner.py`) — the vision LLM (Gemini by default, Ollama as the local option) generates scene/mood descriptions. Also detects faces (Haar cascade) and bodies (HOG). Captions cached as JSON in project dir.
5. **Deduplication** — dHash perceptual hashing removes near-duplicates, keeps sharpest
6. **SemanticMatcher** (`services/matcher.py`) — sentence-transformers encodes mood descriptions + image captions, cosine similarity matrix, Hungarian algorithm for optimal one-to-one assignment. Merges low-energy segments when images < segments.
7. **Location clustering** — Groups GPS-tagged images by place, computes LocationGroups for title cards
8. **VideoGenerator** (`services/video_generator.py`) — Ken Burns effect (1.3x oversized source, cubic-eased zoom/pan), manual crossfade at beat boundaries, smart crop with face detection, location title cards for groups >3 clips. Applies the project's video style (`app/styles`): a per-picture grade plus per-frame bars, vignette and grain. Assembles via moviepy + ffmpeg.

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
- `app/routers/` — upload, project, generate (includes WebSocket at `/ws`), download, llm (`/api/llm/models`), styles (`/api/styles`), copywriting, xhs_publish
- `app/llm/` — provider-neutral LLM layer. `base.py` (the `LLMProvider` interface and errors), `gemini.py`, `ollama.py`, `registry.py` (`Role`, `get_llm`), `parsing.py`, `images.py`
- `app/styles/` — video style plugins. `base.py` (`VideoStyle`, `@register`, registry), `effects.py` (shared image ops, `FrameFinisher`), one module per style
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

- All LLM calls go through `app.llm.get_llm(Role.X).generate(...)`. Never call a vendor API from a service. Roles: `VISION` (captions, crop review), `TEXT` (lyric emotion), `COPYWRITING` (post copy, XHS style analysis)
- Provider choice: `FLOWPIC_LLM_PROVIDER` (default `gemini`) for everything, overridden per role by `FLOWPIC_{VISION,TEXT,COPYWRITING}_PROVIDER`. A project may override the vision provider and model
- `get_llm` wraps the chosen provider in `FallbackProvider` (`app/llm/fallback.py`) when `FLOWPIC_LLM_FALLBACK` (default `ollama`) names a different one. An unusable primary (no key, spent quota) is tried once per job and then skipped; a single failed request falls back for that request only. Tests turn the fallback off in `conftest.py` unless they test it
- `LLMNotConfigured` (missing or rejected key, unknown provider) stops the whole job, and so does its subclass `LLMQuotaExceeded` (a 429 that asks for more than a minute's wait, such as a spent daily limit); any other `LLMError` skips one item
- Captioning runs `provider.parallel_requests` photos at once (Gemini 4, Ollama 1) unless `FLOWPIC_CAPTION_PARALLEL` says otherwise. OpenCV detection is serialised behind a lock because the cascade detector is not thread-safe
- Never run the test suite or a render script while a user render is in progress: both use the GPU
- API keys live only in `backend/.env` (git-ignored). `gemini_api_key` is a `SecretStr`. Never print, log or return a key, and never read `.env` aloud
- The Anthropic API is not used. Do not add it back
- A video style is one file in `app/styles` with a `@register` class; the package imports every module at startup and `/api/styles` lists them. `ProjectConfig.video_style` is a plain string so new styles need no model or frontend change
- Tests in `backend/tests/` are hermetic: `conftest.py` blanks the real key and providers use a mock transport or `FakeLLM`

- GPU models (Music2Emo, Demucs, Whisper) load/unload sequentially to share VRAM
- Image captions are cached in `{project_dir}/captions/{hash}.json` — skip on regenerate
- Uploaded files are cleaned up after video generation; caption cache kept for debugging
- Services use lazy imports inside `video_worker.py` to avoid loading all models at startup
- Config via `pydantic-settings` with `FLOWPIC_` env prefix (e.g., `FLOWPIC_OLLAMA_MODEL`)
- Aspect ratios: 16:9, 21:9, 9:16, 9:19.5 (iPhone full screen), 3:4, 1:1, 4:3. Quality: 720p, 1080p, 2K, 4K. Sizes live in `RESOLUTIONS` in `services/smart_crop.py`; a new ratio also needs the `AspectRatio` enum and the frontend list in `ConfigPanel.tsx`
