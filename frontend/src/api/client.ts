import type {
  CopywritingResult,
  GenerateResponse,
  MusicUploadResponse,
  OllamaModelsResponse,
  ProjectConfig,
  ProjectInfo,
  PublishedImageInfo,
  TaskRecord,
  UploadResponse,
  XhsCookieStatus,
  XhsPublishResult,
  XhsStyleProfile,
} from "../types";

const BASE = "/api";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + url, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

export async function createProject(): Promise<{ project_id: string }> {
  return request("/project/create", { method: "POST" });
}

export async function getProject(projectId: string): Promise<ProjectInfo> {
  return request(`/project/${projectId}`);
}

export async function getActiveProject(): Promise<ProjectInfo | null> {
  const res = await fetch(BASE + "/project/active");
  if (!res.ok || res.status === 204) return null;
  const data = await res.json();
  return data || null;
}

export async function updateConfig(
  projectId: string,
  config: ProjectConfig
): Promise<ProjectConfig> {
  return request(`/project/${projectId}/config`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
}

export async function uploadImages(
  projectId: string,
  files: File[]
): Promise<UploadResponse> {
  const form = new FormData();
  for (const f of files) {
    form.append("files", f);
  }
  return request(`/upload/images?project_id=${projectId}`, {
    method: "POST",
    body: form,
  });
}

export async function uploadMusic(
  projectId: string,
  file: File
): Promise<MusicUploadResponse> {
  const form = new FormData();
  form.append("file", file);
  return request(`/upload/music?project_id=${projectId}`, {
    method: "POST",
    body: form,
  });
}

export async function generateVideo(
  projectId: string
): Promise<GenerateResponse> {
  return request(`/generate/${projectId}`, { method: "POST" });
}

export function getDownloadUrl(taskId: string): string {
  return `${BASE}/download/${taskId}`;
}

export async function fetchTasks(limit = 20, taskType?: string): Promise<TaskRecord[]> {
  let url = `/tasks?limit=${limit}`;
  if (taskType) url += `&task_type=${taskType}`;
  return request(url);
}

export async function cancelTask(taskId: string): Promise<void> {
  return request(`/tasks/${taskId}/cancel`, { method: "POST" });
}

export async function retryTask(taskId: string): Promise<GenerateResponse> {
  return request(`/tasks/${taskId}/retry`, { method: "POST" });
}

export async function fetchOllamaModels(): Promise<OllamaModelsResponse> {
  return request("/ollama/models");
}

export async function cropPreview(
  projectId: string
): Promise<GenerateResponse> {
  return request(`/generate/${projectId}/crop-preview`, { method: "POST" });
}

export async function listCrops(
  projectId: string
): Promise<{ project_id: string; crops: string[] }> {
  return request(`/crops/${projectId}`);
}

export function getCropUrl(projectId: string, filename: string): string {
  return `${BASE}/crops/${projectId}/${filename}`;
}

export function getOriginalImageUrl(projectId: string, cropFilename: string): string {
  // Crop filename format: "003_<stem>_<mode>.png" → extract stem
  const base = cropFilename.replace(/\.[^.]+$/, ""); // strip extension
  const firstUnderscore = base.indexOf("_");
  if (firstUnderscore < 0) return "";
  const rest = base.slice(firstUnderscore + 1); // "<stem>_<mode>"
  const lastUnderscore = rest.lastIndexOf("_");
  if (lastUnderscore < 0) return "";
  const stem = rest.slice(0, lastUnderscore);
  return `${BASE}/images/${projectId}/${stem}`;
}

export async function regenerateCrop(
  projectId: string,
  cropFilename: string,
  feedback: string
): Promise<{ crop_filename: string }> {
  return request(`/crops/${projectId}/regenerate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ crop_filename: cropFilename, feedback }),
  });
}

export async function generateCopywriting(
  projectId: string,
  hint?: string
): Promise<CopywritingResult> {
  return request(`/copywriting/${projectId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ hint: hint || "" }),
  });
}

export async function getCopywriting(
  projectId: string
): Promise<CopywritingResult> {
  return request(`/copywriting/${projectId}`);
}

// --- XHS Publishing ---

export async function getXhsCookieStatus(): Promise<XhsCookieStatus> {
  return request("/xhs/cookies");
}

export async function saveXhsCookies(cookie: string): Promise<XhsCookieStatus> {
  return request("/xhs/cookies", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cookie }),
  });
}

export async function clearXhsCookies(): Promise<void> {
  return request("/xhs/cookies", { method: "DELETE" });
}

export async function publishToXhs(
  projectId: string,
  data: { title: string; description: string; hashtags: string[]; image_filenames: string[] }
): Promise<XhsPublishResult> {
  return request(`/xhs/publish/${projectId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

// --- XHS Style Scanner ---

export async function scanXhsStyleProfile(force = false): Promise<XhsStyleProfile> {
  return request(`/xhs/style-profile?force=${force}`, { method: "POST" });
}

export async function getXhsStyleProfile(): Promise<XhsStyleProfile> {
  return request("/xhs/style-profile");
}

export async function clearXhsStyleProfile(): Promise<void> {
  return request("/xhs/style-profile", { method: "DELETE" });
}

// --- Published Images ---

export async function getPublishedImages(
  projectId: string
): Promise<{ project_id: string; published_images: PublishedImageInfo[] }> {
  return request(`/xhs/published-images/${projectId}`);
}

export async function removePublishedImages(
  projectId: string
): Promise<{ removed: number }> {
  return request(`/project/${projectId}/published-images`, { method: "DELETE" });
}
