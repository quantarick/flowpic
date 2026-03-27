import { useCallback, useEffect, useState } from "react";
import * as api from "../api/client";
import type { AspectRatio, ProjectConfig, Quality, TaskStatus } from "../types";

const STORAGE_KEY = "flowpic_session";

function loadSession(): { projectId: string | null; taskId: string | null } {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch {}
  return { projectId: null, taskId: null };
}

function saveSession(projectId: string | null, taskId: string | null) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ projectId, taskId }));
  } catch {}
}

const TERMINAL_STATUSES = new Set(["done", "failed", "cancelled"]);

export interface ProjectState {
  projectId: string | null;
  images: string[];
  music: string | null;
  musicDuration: number | null;
  config: ProjectConfig;
  taskId: string | null;
  status: TaskStatus | null;
  videoUrl: string | null;
  error: string | null;
  loading: boolean;
}

const defaultConfig: ProjectConfig = {
  aspect_ratio: "16:9",
  quality: "720p",
  fps: 30,
  vision_model: null,
  skip_crop_review: true,
};

export function useProject() {
  const [state, setState] = useState<ProjectState>({
    projectId: null,
    images: [],
    music: null,
    musicDuration: null,
    config: defaultConfig,
    taskId: null,
    status: null,
    videoUrl: null,
    error: null,
    loading: true, // loading while checking for active project
  });

  // Restore project and active task on mount
  useEffect(() => {
    (async () => {
      try {
        const saved = loadSession();

        // Ask backend for the active project (has uploads, no completed task)
        const info = await api.getActiveProject();
        if (!info) {
          // No active project — check if saved task is done
          if (saved.projectId && saved.taskId) {
            const tasks = await api.fetchTasks(50);
            const doneTask = tasks.find((t) => t.task_id === saved.taskId);
            if (doneTask?.status === "done") {
              const projInfo = await api.getProject(saved.projectId);
              setState((s) => ({
                ...s,
                projectId: saved.projectId,
                images: projInfo.images,
                music: projInfo.music,
                config: { ...defaultConfig, ...projInfo.config },
                taskId: doneTask.task_id,
                status: "done",
                videoUrl: api.getDownloadUrl(doneTask.task_id),
                loading: false,
              }));
              return;
            }
          }
          saveSession(null, null);
          setState((s) => ({ ...s, projectId: null, loading: false }));
          return;
        }

        const baseState: Partial<ProjectState> = {
          projectId: info.project_id,
          images: info.images,
          music: info.music,
          config: { ...defaultConfig, ...info.config },
          loading: false,
        };

        // Check if this project has an active task to reconnect to
        const tasks = await api.fetchTasks(50);
        const activeTask = tasks.find(
          (t) => t.project_id === info.project_id && !TERMINAL_STATUSES.has(t.status)
        );

        if (activeTask) {
          baseState.taskId = activeTask.task_id;
          baseState.status = activeTask.status;
          saveSession(info.project_id, activeTask.task_id);
        } else {
          saveSession(info.project_id, null);
        }

        setState((s) => ({ ...s, ...baseState }));
      } catch {
        saveSession(null, null);
        setState((s) => ({ ...s, projectId: null, loading: false }));
      }
    })();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const createProject = useCallback(async () => {
    setState((s) => ({ ...s, loading: true, error: null }));
    try {
      const { project_id } = await api.createProject();
      saveSession(project_id, null);
      setState((s) => ({ ...s, projectId: project_id, loading: false }));
      return project_id;
    } catch (e) {
      setState((s) => ({
        ...s,
        loading: false,
        error: (e as Error).message,
      }));
      return null;
    }
  }, []);

  const uploadImages = useCallback(
    async (files: File[]) => {
      let pid = state.projectId;
      if (!pid) {
        pid = await createProject();
        if (!pid) return;
      }
      setState((s) => ({ ...s, loading: true, error: null }));
      try {
        const res = await api.uploadImages(pid, files);
        setState((s) => ({
          ...s,
          projectId: pid,
          images: [...s.images, ...res.filenames],
          loading: false,
        }));
      } catch (e) {
        setState((s) => ({
          ...s,
          loading: false,
          error: (e as Error).message,
        }));
      }
    },
    [state.projectId, createProject]
  );

  const uploadMusic = useCallback(
    async (file: File) => {
      let pid = state.projectId;
      if (!pid) {
        pid = await createProject();
        if (!pid) return;
      }
      setState((s) => ({ ...s, loading: true, error: null }));
      try {
        const res = await api.uploadMusic(pid, file);
        setState((s) => ({
          ...s,
          projectId: pid,
          music: res.filename,
          musicDuration: res.duration_seconds,
          loading: false,
        }));
      } catch (e) {
        setState((s) => ({
          ...s,
          loading: false,
          error: (e as Error).message,
        }));
      }
    },
    [state.projectId, createProject]
  );

  const updateConfig = useCallback(
    async (updates: Partial<ProjectConfig>) => {
      const newConfig = { ...state.config, ...updates };
      setState((s) => ({ ...s, config: newConfig }));
      if (state.projectId) {
        await api.updateConfig(state.projectId, newConfig);
      }
    },
    [state.projectId, state.config]
  );

  const generate = useCallback(async () => {
    if (!state.projectId) return;
    setState((s) => ({ ...s, loading: true, error: null, videoUrl: null }));
    try {
      // Ensure config is synced to server before generating
      await api.updateConfig(state.projectId, state.config);
      const { task_id } = await api.generateVideo(state.projectId);
      saveSession(state.projectId, task_id);
      setState((s) => ({
        ...s,
        taskId: task_id,
        status: "pending",
        loading: false,
      }));
    } catch (e) {
      setState((s) => ({
        ...s,
        loading: false,
        error: (e as Error).message,
      }));
    }
  }, [state.projectId, state.config]);

  const cropPreview = useCallback(async () => {
    if (!state.projectId) return;
    setState((s) => ({ ...s, loading: true, error: null, videoUrl: null }));
    try {
      await api.updateConfig(state.projectId, state.config);
      const { task_id } = await api.cropPreview(state.projectId);
      setState((s) => ({
        ...s,
        taskId: task_id,
        status: "pending",
        loading: false,
      }));
    } catch (e) {
      setState((s) => ({
        ...s,
        loading: false,
        error: (e as Error).message,
      }));
    }
  }, [state.projectId, state.config]);

  const setStatus = useCallback((status: TaskStatus) => {
    setState((s) => {
      const updated = { ...s, status };
      if (status === "done" && s.taskId) {
        updated.videoUrl = api.getDownloadUrl(s.taskId);
      }
      return updated;
    });
  }, []);

  const setDone = useCallback((taskId: string) => {
    setState((s) => {
      saveSession(s.projectId, taskId);
      return {
        ...s,
        status: "done",
        taskId,
        videoUrl: api.getDownloadUrl(taskId),
      };
    });
  }, []);

  const setTaskId = useCallback((taskId: string) => {
    setState((s) => {
      saveSession(s.projectId, taskId);
      return {
        ...s,
        taskId,
        status: "pending",
        videoUrl: null,
        error: null,
      };
    });
  }, []);

  const refreshProject = useCallback(async () => {
    if (!state.projectId) return;
    try {
      const info = await api.getProject(state.projectId);
      setState((s) => ({
        ...s,
        images: info.images,
        music: info.music,
        config: { ...defaultConfig, ...info.config },
      }));
    } catch {}
  }, [state.projectId]);

  return {
    ...state,
    createProject,
    uploadImages,
    uploadMusic,
    updateConfig,
    generate,
    cropPreview,
    setStatus,
    setDone,
    setTaskId,
    refreshProject,
  };
}
