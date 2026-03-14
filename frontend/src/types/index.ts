export type AspectRatio = "16:9" | "21:9" | "9:16" | "1:1" | "4:3";
export type Quality = "720p" | "1080p" | "2k" | "4k";
export type TaskStatus =
  | "pending"
  | "analyzing_audio"
  | "analyzing_lyrics"
  | "classifying_emotion"
  | "captioning_images"
  | "matching"
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
  created_at: string;
  finished_at: string | null;
  output_path: string | null;
  image_count: number;
  duration_seconds: number | null;
  config: ProjectConfig | null;
  error_message: string | null;
}
