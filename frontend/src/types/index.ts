export type AspectRatio = "16:9" | "21:9" | "9:16" | "1:1" | "4:3";
export type Quality = "720p" | "1080p" | "2k" | "4k";
export type TaskStatus =
  | "pending"
  | "analyzing_audio"
  | "analyzing_lyrics"
  | "classifying_emotion"
  | "captioning_images"
  | "matching"
  | "reviewing_crops"
  | "rendering"
  | "encoding"
  | "done"
  | "failed"
  | "cancelled";

export interface ProjectConfig {
  aspect_ratio: AspectRatio;
  quality: Quality;
  fps: number;
  vision_model: string | null;
  skip_crop_review: boolean;
}

export interface OllamaModel {
  name: string;
  size: number | null;
  parameter_size: string | null;
}

export interface OllamaModelsResponse {
  default: string;
  models: OllamaModel[];
}

export interface ProjectInfo {
  project_id: string;
  images: string[];
  music: string | null;
  config: ProjectConfig;
  created_at: string;
}

export interface ProgressMessage {
  status: TaskStatus;
  progress: number;
  current_step: string;
  detail: string;
  eta_seconds: number | null;
}

export interface UploadResponse {
  filenames: string[];
  count: number;
}

export interface MusicUploadResponse {
  filename: string;
  duration_seconds: number | null;
}

export interface GenerateResponse {
  task_id: string;
}

export interface TaskRecord {
  task_id: string;
  project_id: string;
  status: TaskStatus;
  task_type?: string;
  created_at: string;
  finished_at: string | null;
  output_path: string | null;
  image_count: number;
  duration_seconds: number | null;
  config: ProjectConfig | null;
  error_message: string | null;
}

export interface CopywritingResult {
  title: string;
  description: string;
  hashtags: string[];
  cover_index: number;
}

export interface XhsCookieStatus {
  connected: boolean;
  username: string | null;
  user_id: string | null;
  expired: boolean;
  error: string | null;
}

export interface XhsPublishResult {
  success: boolean;
  post_url: string | null;
  note_id: string | null;
  error: string | null;
}

export interface ScrapedPost {
  title: string;
  description: string;
  hashtags: string[];
  note_url: string | null;
}

export interface XhsStyleProfile {
  tone: string;
  emoji_style: string;
  sentence_structure: string;
  hashtag_strategy: string;
  title_pattern: string;
  sample_phrases: string[];
  overall_summary: string;
  scraped_posts: ScrapedPost[];
  scraped_at: string;
  error: string | null;
}

export interface PublishedImageInfo {
  crop_filename: string;
  image_hash: string;
  crop_mode: string;
  published_at: string;
  post_url: string | null;
  note_id: string | null;
}
