# FlowPic

Music-driven photo slideshow generator. Upload photos and a song, and FlowPic automatically creates a cinematic video — like iPhone Memories — with Ken Burns effects, beat-synced transitions, and AI-powered image-to-music matching.

## How it works

FlowPic analyzes your music and photos separately, then intelligently pairs them:

1. **Audio analysis** — Extracts beats, tempo, energy, and spectral features using librosa. Segments the song at 4-beat (one measure) boundaries.
2. **Music emotion** — Runs [Music2Emo](https://github.com/AMAAI-Lab/Music2Emotion) to classify valence/arousal per segment, producing natural language mood descriptions like *"Energetic, joyful, and uplifting"*.
3. **Lyrics analysis** (optional) — Separates vocals with [Demucs](https://github.com/facebookresearch/demucs), transcribes with [Whisper](https://github.com/openai/whisper), and extracts per-segment lyric emotions to enrich the mood descriptions.
4. **Image captioning** — A local vision-language model ([moondream](https://moondream.ai/) via [Ollama](https://ollama.com/)) describes each photo's scene, mood, and visual qualities.
5. **Semantic matching** — Computes cosine similarity between music mood embeddings and image caption embeddings using [sentence-transformers](https://www.sbert.net/), then finds the optimal assignment via the Hungarian algorithm. Each image is used at most once; if the song has more segments than images, low-energy segments are merged.
6. **GPS clustering** — Photos with EXIF GPS data are grouped by location. Consecutive clips from the same place are clustered together, and groups with 4+ images get a cinematic title card (blurred/darkened backdrop with location text).
7. **Video rendering** — Each image gets a Ken Burns effect (smooth zoom + pan with cubic easing), transitions crossfade at beat boundaries, and the whole thing is encoded to MP4 with the original audio.

## Features

- Drag-and-drop web UI for images and music
- AI-powered image↔music matching (semantic, not just color/brightness)
- Ken Burns zoom/pan with face-aware centering
- Beat-synced crossfade transitions
- Configurable aspect ratio (16:9, 21:9, 9:16, 1:1, 4:3) with smart cropping
- Output quality up to 4K
- GPS-based location title cards
- Near-duplicate image detection (dHash)
- Real-time progress via WebSocket with cancel support
- Task history with retry
- i18n (English, Chinese)

## Prerequisites

- **Python 3.11+**
- **Node.js 20+**
- **FFmpeg 6+** — required by moviepy (`winget install ffmpeg` on Windows)
- **Ollama** — for local image captioning (`ollama pull moondream`)
- **NVIDIA GPU** with 8+ GB VRAM recommended (for Music2Emo, Demucs, Whisper, Ollama)

## Setup

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
```

### Ollama

```bash
ollama pull moondream
ollama serve  # if not running as a service
```

## Running

Start the backend and frontend in separate terminals:

```bash
# Terminal 1: Backend
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

Open http://localhost:5173 in your browser.

### Docker Compose

```bash
docker compose up
```

This starts the backend (port 8000), frontend (port 5173), and Ollama (port 11434) with GPU passthrough.

## Configuration

All settings can be overridden via environment variables with the `FLOWPIC_` prefix:

| Variable | Default | Description |
|---|---|---|
| `FLOWPIC_DATA_DIR` | `data` | Project data directory |
| `FLOWPIC_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `FLOWPIC_OLLAMA_MODEL` | `moondream` | Vision model for captioning |
| `FLOWPIC_LYRICS_ENABLED` | `true` | Enable vocal separation + lyrics analysis |
| `FLOWPIC_WHISPER_MODEL_SIZE` | `medium` | Whisper model size (tiny/base/small/medium/large) |
| `FLOWPIC_MAX_IMAGES` | `100` | Max images per project |
| `FLOWPIC_MAX_WORKERS` | `2` | Concurrent generation tasks |
| `FLOWPIC_CAPTION_PARALLEL` | `3` | Parallel image captioning threads |

## Tech Stack

**Backend**: FastAPI, librosa, Music2Emo, Demucs, Whisper, sentence-transformers, OpenCV, moviepy, geopy

**Frontend**: React 18, TypeScript, Vite

**Infrastructure**: Ollama (moondream), Docker Compose, NVIDIA GPU

## License

MIT
