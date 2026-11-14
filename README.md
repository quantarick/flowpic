# FlowPic

Music-driven photo slideshow generator. Upload photos and a song, and FlowPic automatically creates a cinematic video — like iPhone Memories — with Ken Burns effects, beat-synced transitions, and AI-powered image-to-music matching.

## How it works

FlowPic analyzes your music and photos separately, then intelligently pairs them:

1. **Audio analysis** — Extracts beats, tempo, energy, and spectral features using librosa. Segments the song at 4-beat (one measure) boundaries.
2. **Music emotion** — Runs [Music2Emo](https://github.com/AMAAI-Lab/Music2Emotion) to classify valence/arousal per segment, producing natural language mood descriptions like *"Energetic, joyful, and uplifting"*.
3. **Lyrics analysis** (optional) — Separates vocals with [Demucs](https://github.com/facebookresearch/demucs), transcribes with [Whisper](https://github.com/openai/whisper), and extracts per-segment lyric emotions to enrich the mood descriptions.
4. **Image captioning** — A vision model describes each photo's scene, mood, and visual qualities. [Gemini](https://ai.google.dev/) is the default; a local model through [Ollama](https://ollama.com/) is the alternative. See [LLM providers](#llm-providers).
5. **Semantic matching** — Computes cosine similarity between music mood embeddings and image caption embeddings using [sentence-transformers](https://www.sbert.net/), then finds the optimal assignment via the Hungarian algorithm. Each image is used at most once; if the song has more segments than images, low-energy segments are merged.
6. **GPS clustering** — Photos with EXIF GPS data are grouped by location. Consecutive clips from the same place are clustered together, and groups with 4+ images get a cinematic title card (blurred/darkened backdrop with location text).
7. **Video rendering** — Each image gets a Ken Burns effect (smooth zoom + pan with cubic easing), transitions crossfade at beat boundaries, and the whole thing is encoded to MP4 with the original audio. A [video style](#video-styles) sets the look of the pictures.

## Features

- Drag-and-drop web UI for images and music
- Folder import — drop or pick a folder and every JPG, PNG and WebP in it and its sub-folders is added
- Reset — one button deletes every project and all history, after showing what will go
- AI-powered image↔music matching (semantic, not just color/brightness)
- Ken Burns zoom/pan with face-aware centering
- Beat-synced crossfade transitions
- Configurable aspect ratio (16:9, 21:9, 4:3, 1:1, and vertical 9:16, 3:4 and 9:19.5 for a full iPhone screen) with smart cropping
- Output quality up to 4K
- GPS-based location title cards
- Near-duplicate image detection (dHash)
- Real-time progress via WebSocket with cancel support
- Task history with retry
- Video styles as plugins — natural, art film, street photo and vintage film, with room for your own
- Pluggable LLM providers — Gemini (hosted) or Ollama (local), chosen per task
- AI copywriting — generates XHS-style titles, descriptions, and hashtags from image captions
- One-click XHS (Xiaohongshu) publishing — automated browser upload via Playwright with cookie-based auth
- i18n (English, Chinese)

## Prerequisites

- **Python 3.11+**
- **Node.js 20+**
- **FFmpeg 6+** — required by moviepy (`winget install ffmpeg` on Windows)
- **A Gemini API key** — for captioning, lyric emotion and copywriting with the default provider
- **Ollama** (optional) — only if you choose the local provider (`ollama pull llava-phi3`)
- **NVIDIA GPU** with 8+ GB VRAM recommended (for Music2Emo, Demucs, Whisper, Ollama)
- **Playwright** (optional) — for XHS publishing (`playwright install chromium`)

## Setup

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env   # then put your Gemini API key in .env
```

`.env` is git-ignored. Keep keys there and nowhere else.

### Frontend

```bash
cd frontend
npm install
```

### Ollama (optional, local provider only)

```bash
ollama pull llava-phi3
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
| `FLOWPIC_LLM_PROVIDER` | `gemini` | LLM for every task: `gemini` or `ollama` |
| `FLOWPIC_LLM_FALLBACK` | `ollama` | Provider used when the chosen one cannot answer. Empty turns the fallback off |
| `FLOWPIC_VISION_PROVIDER` | — | Override for image captioning and crop review |
| `FLOWPIC_TEXT_PROVIDER` | — | Override for lyric emotion |
| `FLOWPIC_COPYWRITING_PROVIDER` | — | Override for copywriting and XHS style analysis |
| `FLOWPIC_GEMINI_API_KEY` | — | Gemini API key. Set it in `backend/.env` only |
| `FLOWPIC_GEMINI_MODEL` | `gemini-4-hydrogen` | Gemini model |
| `FLOWPIC_GEMINI_THINKING_LEVEL` | `low` | Gemini thinking level; empty uses the model's default |
| `FLOWPIC_GEMINI_TIMEOUT` | `120` | Seconds to wait for one Gemini reply |
| `FLOWPIC_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `FLOWPIC_OLLAMA_MODEL` | `llava-phi3` | Ollama model |
| `FLOWPIC_LYRICS_ENABLED` | `true` | Enable vocal separation + lyrics analysis |
| `FLOWPIC_WHISPER_MODEL_SIZE` | `medium` | Whisper model size (tiny/base/small/medium/large) |
| `FLOWPIC_MAX_IMAGES` | `100` | Max images per project |
| `FLOWPIC_MAX_WORKERS` | `2` | Concurrent generation tasks |
| `FLOWPIC_CAPTION_PARALLEL` | `0` | Photos captioned at once. `0` picks for the provider: 1 for a local model, 4 for Gemini |
| `FLOWPIC_CAPTION_MAX_TOKENS` | `450` | Longest caption reply for a local model, so one that ignores the format cannot run on. `0` removes the cap |
| `FLOWPIC_XHS_HEADLESS` | `true` | Run Playwright browser in headless mode for XHS publish |

## LLM providers

Every model call goes through `backend/app/llm`. Services ask for a provider by role (vision, text, copywriting) and call `generate`; they never touch a vendor API.

- **gemini** (default) — Google's Interactions API. Needs `FLOWPIC_GEMINI_API_KEY`.
- **ollama** — a local model. Needs a running Ollama server and no key.

`FLOWPIC_LLM_PROVIDER` picks the provider for everything. A per-role variable overrides it, so `FLOWPIC_VISION_PROVIDER=ollama` keeps photos on your machine while copywriting still uses Gemini. A project can also pick its own vision model from the dropdown in the UI.

If Gemini cannot answer, FlowPic falls back to the local Ollama model so the job keeps going. A missing key or a spent daily quota switches the rest of the job to the local model; a single failed request sends just that request there. The progress line says when this has happened. The local model gets one request at a time. Set `FLOWPIC_LLM_FALLBACK=` to turn this off, in which case a spent quota stops the job with Google's message.

What the Gemini provider sends: the prompt, and for vision tasks each photo shrunk to 2048px on its longer side and re-encoded, which removes EXIF data including GPS. Requests are sent with `store: false`. The API key travels in a request header, is masked in logs and is never returned by any endpoint.

To add a provider, subclass `LLMProvider` in a new module under `backend/app/llm` and add it to `PROVIDERS` in `registry.py`.

## Video styles

A video style sets how pictures look in the render. Pick one in the UI before generating.

| Style | Look |
|---|---|
| `natural` | Pictures as shot. The default. |
| `art_film` | Muted colour, faded blacks, teal shadows, 2.39:1 bars, fine grain |
| `street` | High-contrast black and white with heavy grain |
| `vintage` | Warm, faded colour with soft contrast |

Styles are plugins. Each one is a single file in `backend/app/styles`, and the UI lists whatever is there. To add one, create a file such as `backend/app/styles/noir.py`:

```python
import numpy as np

from app.styles import effects as fx
from app.styles.base import VideoStyle, register


@register
class Noir(VideoStyle):
    name = "noir"
    label = {"en": "Noir", "zh": "黑色电影"}
    description = {"en": "Deep blacks and hard light.", "zh": "浓重的暗部与硬光。"}

    letterbox = 2.39   # cinema bars on wide outputs; omit to fill the frame
    vignette = 0.4     # corner darkening, 0 to 1
    grain = 8.0        # animated film grain strength

    def grade(self, image: np.ndarray) -> np.ndarray:
        return fx.tone_curve(fx.monochrome(image), contrast=1.6)
```

Restart the backend and the style appears in the dropdown. `grade` runs once per picture. Bars, vignette and grain are applied to every frame, so they stay fixed while the picture moves. Grain makes files larger.

## Tech Stack

**Backend**: FastAPI, librosa, Music2Emo, Demucs, Whisper, sentence-transformers, OpenCV, moviepy, geopy, Playwright, Google Gemini

**Frontend**: React 18, TypeScript, Vite

**Infrastructure**: Gemini API or Ollama, Docker Compose, NVIDIA GPU

## License

MIT
