const en = {
  // App
  appTitle: "FlowPic",
  appTagline: "Music-driven photo slideshow generator",
  sectionImages: "1. Upload Images",
  sectionMusic: "2. Upload Music",
  sectionConfig: "3. Configure",
  sectionResult: "Result",

  // ImageUploader
  imgDropHint: "Drop images here or click to browse",
  imgFormatHint: "JPG, PNG, WebP - up to 100 images, 20MB each",
  imgUploaded: (n: number) => `${n} image${n !== 1 ? "s" : ""} uploaded`,
  imgAddMore: "Click or drop to add more",

  // MusicUploader
  musicDropHint: "Drop a music file here or click to browse",
  musicFormatHint: "MP3, WAV, FLAC, M4A - max 100MB, 10 min",
  musicDuration: "Duration:",
  musicReplace: "Click or drop to replace",

  // ConfigPanel
  cfgAspectRatio: "Aspect Ratio:",
  cfgQuality: "Quality:",
  cfgVisionModel: "Vision Model:",
  cfgFps: "FPS:",
  arLandscape: "16:9 (Landscape)",
  arUltrawide: "21:9 (Ultrawide)",
  arPortrait: "9:16 (Portrait)",
  arSquare: "1:1 (Square)",
  arClassic: "4:3 (Classic)",
  q720: "720p (SD)",
  q1080: "1080p (HD)",
  q2k: "2K (QHD)",
  q4k: "4K (UHD)",

  // GenerateButton
  btnStarting: "Starting...",
  btnGenerate: "Generate Video",

  // ProgressBar
  stepPending: "Queued...",
  stepAnalyzingAudio: "Analyzing audio",
  stepAnalyzingLyrics: "Analyzing lyrics",
  stepClassifyingEmotion: "Classifying emotion",
  stepCaptioning: "Captioning images",
  stepMatching: "Matching images to music",
  stepRendering: "Rendering video",
  stepEncoding: "Encoding video",
  stepDone: "Complete!",
  stepFailed: "Failed",
  stepCancelled: "Cancelled",
  btnCancel: "Cancel",

  // VideoPreview
  btnDownload: "Download Video",

  // TaskHistory
  taskHistory: "Task History",
  statusPending: "Pending",
  statusAnalyzing: "Analyzing",
  statusProcessing: "Processing",
  statusMatching: "Matching",
  statusRendering: "Rendering",
  statusEncoding: "Encoding",
  statusDone: "Done",
  statusFailed: "Failed",
  statusCancelled: "Cancelled",
  taskImages: (n: number) => `${n} images`,
  timeJustNow: "just now",
  timeMinAgo: (m: number) => `${m}m ago`,
  timeHrAgo: (h: number) => `${h}h ago`,
  btnStop: "Stop",
  btnRetry: "Retry",
  retryFailed: (msg: string) => `Retry failed: ${msg}`,
} as const;

// Widen literal types so translations can use different string values
type Widen<T> = {
  [K in keyof T]: T[K] extends string
    ? string
    : T[K] extends (...args: infer A) => string
      ? (...args: A) => string
      : T[K];
};

export type Translations = Widen<typeof en>;
export default en;
