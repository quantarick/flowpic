import type {
  GenerateResponse,
  MusicUploadResponse,
  OllamaModelsResponse,
  ProjectConfig,
  ProjectInfo,
  TaskRecord,
  UploadResponse,
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

export async function fetchTasks(limit = 20): Promise<TaskRecord[]> {
  return request(`/tasks?limit=${limit}`);
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
