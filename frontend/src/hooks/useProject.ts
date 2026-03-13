import { useCallback, useState } from "react";
import * as api from "../api/client";
import type { AspectRatio, ProjectConfig, Quality, TaskStatus } from "../types";

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
    loading: false,
  });

  const createProject = useCallback(async () => {
    setState((s) => ({ ...s, loading: true, error: null }));
    try {
      const { project_id } = await api.createProject();
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
      const { task_id } = await api.generateVideo(state.projectId);
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
  }, [state.projectId]);

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
    setState((s) => ({
      ...s,
      status: "done",
      taskId,
      videoUrl: api.getDownloadUrl(taskId),
    }));
  }, []);

  const setTaskId = useCallback((taskId: string) => {
    setState((s) => ({
      ...s,
      taskId,
      status: "pending",
      videoUrl: null,
      error: null,
    }));
  }, []);

  return {
    ...state,
    createProject,
    uploadImages,
    uploadMusic,
    updateConfig,
    generate,
    setStatus,
    setDone,
    setTaskId,
  };
}
