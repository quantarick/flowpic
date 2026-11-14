const en = {
  // App
  appTitle: "FlowPic",
  appTagline: "Music-driven photo slideshow generator",
  sectionImages: "Upload Images",
  sectionMusic: "Upload Music",
  sectionConfig: "Configure",
  sectionResult: "Result",
  tabCrops: "Crops",
  tabVideo: "Video",
  hintPreviewCrops: "Tip: Preview crops first in the Crops tab to review framing before generating.",
  btnNewProject: "New Project",

  // Reset
  btnReset: "Reset",
  resetTitle: "Delete everything?",
  resetCounting: "Counting what is stored…",
  resetNothing: "Nothing is stored, so there is nothing to delete.",
  resetSummary: (projects: number, photos: number, songs: number, videos: number, tasks: number, size: string) =>
    `This deletes ${projects} project${projects !== 1 ? "s" : ""}: ${photos} photos, ${songs} songs, ${videos} rendered videos, every caption and cache, and ${tasks} history entries. ${size} in total.`,
  resetWarning: "This cannot be undone.",
  resetKept: "Your Xiaohongshu login and style profile are kept. The record of photos already published is cleared.",
  resetConfirm: "Delete everything",
  resetCancel: "Cancel",
  resetClose: "Close",
  resetWorking: "Deleting…",
  resetBusy: "A task is running. Cancel it or wait for it to finish first.",
  resetPartly: (n: number) =>
    `${n} project${n !== 1 ? "s" : ""} could not be fully deleted because a file is in use. Close any open video and try again.`,
  resetFailed: (msg: string) => `Reset failed: ${msg}`,

  // ImageUploader
  imgDropHint: "Drop images or a folder here, or click to browse",
  imgPickFolder: "Import a folder",
  imgScanning: "Reading the folder…",
  imgUploadProgress: (done: number, total: number) => `Uploading ${done} of ${total}…`,
  imgImported: (n: number, folders: number) =>
    folders > 0
      ? `Added ${n} image${n !== 1 ? "s" : ""} from ${folders} folder${folders !== 1 ? "s" : ""}.`
      : `Added ${n} image${n !== 1 ? "s" : ""}.`,
  imgNothingFound: "No JPG, PNG or WebP images were found.",
  imgSkippedType: (n: number, kinds: string) =>
    `Skipped ${n} file${n !== 1 ? "s" : ""} that ${n !== 1 ? "are" : "is"} not JPG, PNG or WebP (${kinds}).`,
  imgSkippedLarge: (n: number) => `Skipped ${n} image${n !== 1 ? "s" : ""} over 50MB.`,
  imgLeftOut: (n: number, max: number) => `Left out ${n}: a project holds ${max} images.`,
  imgFormatHint: "JPG, PNG, WebP - up to 100 images, 50MB each",
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
  cfgNoApiKey: "no API key",
  cfgVideoStyle: "Video Style:",
  cfgFps: "FPS:",
  cfgFitStrategy: "Fit Mode:",
  fitCrop: "Crop",
  fitFrame: "Frame",
  cfgFrameStyle: "Frame Style:",
  frameFilm: "Film",
  frameClean: "Clean",
  framePolaroid: "Polaroid",
  frameShadow: "Shadow",
  arLandscape: "16:9 (Landscape)",
  arUltrawide: "21:9 (Ultrawide)",
  arPortrait: "9:16 (Portrait)",
  arPhone: "9:19.5 (iPhone full screen)",
  arPortraitPhoto: "3:4 (Vertical photo)",
  arSquare: "1:1 (Square)",
  arClassic: "4:3 (Classic)",
  q720: "720p (SD)",
  q1080: "1080p (HD)",
  q2k: "2K (QHD)",
  q4k: "4K (UHD)",

  // GenerateButton
  btnStarting: "Starting...",
  btnGenerate: "Generate Video",
  btnCropPreview: "Preview Crops",

  // CropPreview
  cropPreviewTitle: "Crop Preview",
  cropLoading: "Loading crops...",
  cropEmpty: "No crops generated yet",
  cropFeedbackPlaceholder: "What's wrong? e.g. focus on the sheep, not the fence",
  btnRegenerate: "Regenerate",
  regenerating: "Regenerating...",

  // ProgressBar
  stepPending: "Queued...",
  stepAnalyzingAudio: "Analyzing audio",
  stepAnalyzingLyrics: "Analyzing lyrics",
  stepClassifyingEmotion: "Classifying emotion",
  stepCaptioning: "Captioning images",
  stepMatching: "Matching images to music",
  stepReviewingCrops: "Reviewing crops",
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
  statusReviewing: "Reviewing",
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
  btnContinue: "Continue",
  retryFailed: (msg: string) => `Retry failed: ${msg}`,

  // Publish / Copywriting
  tabPublish: "Publish",
  btnGenerateCopy: "Generate Copy",
  btnGeneratingCopy: "Generating...",
  copyTitle: "Title",
  copyDesc: "Description",
  copyHashtags: "Hashtags",
  copyCover: "Publish Images",
  btnSelectAll: "Select All",
  btnDeselectAll: "Deselect All",
  btnCopy: "Copy",
  copiedMsg: "Copied!",
  copyHint: "Style Hint",
  copyHintPlaceholder: "e.g. playful tone, focus on food, mention sunset...",
  copyEmpty: "Click \"Generate Copy\" to create Xiaohongshu copywriting from your photos.",
  copyError: (msg: string) => `Generation failed: ${msg}`,
  btnPreview: "Preview",
  btnEdit: "Edit",
  publishCrops: "Your Crops",
  publishVideo: "Your Video",

  // XHS
  xhsAccount: "XHS Account",
  xhsConnect: "Connect",
  xhsDisconnect: "Disconnect",
  xhsCancel: "Cancel",
  xhsConnecting: "Connecting...",
  xhsSaveCookies: "Save Cookies",
  xhsCookieHint: "Use the cookies.txt Chrome extension to export xiaohongshu.com cookies as JSON, then paste the JSON array here.",
  xhsExpired: "Cookie expired — please reconnect your account.",
  xhsBtnPublish: "Publish to XHS",
  xhsPublishing: "Publishing...",
  xhsPublishSuccess: "Published!",
  xhsViewPost: "View Post",
  xhsPublishFailed: (msg: string) => `Publish failed: ${msg}`,

  // Style Scanner
  xhsScanStyle: "Scan My Style",
  xhsScanning: "Scanning...",
  xhsStyleActive: "Style profile active",
  xhsRescan: "Re-scan",
  xhsClearStyle: "Clear",
  xhsStyleError: (msg: string) => `Style scan failed: ${msg}`,

  // Published Images
  publishedBadge: "Published",
  btnCleanPublished: "Clean Up Published",
  cleanedMsg: (n: number) => `Removed ${n} published image${n !== 1 ? "s" : ""}`,
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
