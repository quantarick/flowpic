import { useCallback, useEffect, useState } from "react";
import * as api from "../api/client";
import { useI18n } from "../i18n";
import type { TaskRecord } from "../types";

interface Props {
  activeTaskId: string | null;
  onSelect: (taskId: string) => void;
  onRetry: (newTaskId: string) => void;
}

export function TaskHistory({ activeTaskId, onSelect, onRetry }: Props) {
  const { t } = useI18n();
  const [tasks, setTasks] = useState<TaskRecord[]>([]);
  const [collapsed, setCollapsed] = useState(false);
  const [retrying, setRetrying] = useState<string | null>(null);
  const [stopping, setStopping] = useState<string | null>(null);

  const STATUS_LABELS: Record<string, { label: string; color: string }> = {
    pending: { label: t.statusPending, color: "#f39c12" },
    analyzing_audio: { label: t.statusAnalyzing, color: "#3498db" },
    analyzing_lyrics: { label: t.statusAnalyzing, color: "#3498db" },
    classifying_emotion: { label: t.statusAnalyzing, color: "#3498db" },
    captioning_images: { label: t.statusProcessing, color: "#3498db" },
    matching: { label: t.statusMatching, color: "#3498db" },
    rendering: { label: t.statusRendering, color: "#9b59b6" },
    encoding: { label: t.statusEncoding, color: "#9b59b6" },
    done: { label: t.statusDone, color: "#2ecc71" },
    failed: { label: t.statusFailed, color: "#e74c3c" },
    cancelled: { label: t.statusCancelled, color: "#95a5a6" },
  };

  function formatDate(iso: string): string {
    const d = new Date(iso);
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    const diffMin = Math.floor(diffMs / 60000);
    if (diffMin < 1) return t.timeJustNow;
    if (diffMin < 60) return t.timeMinAgo(diffMin);
    const diffHr = Math.floor(diffMin / 60);
    if (diffHr < 24) return t.timeHrAgo(diffHr);
    return d.toLocaleDateString();
  }

  function formatDuration(seconds: number): string {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return `${m}m ${s}s`;
  }

  const refresh = useCallback(async () => {
    try {
      const list = await api.fetchTasks(20);
      setTasks(list);
    } catch {
      // silently ignore
    }
  }, []);

  useEffect(() => {
    refresh();
    const hasActive = tasks.some(
      (tk) => !["done", "failed", "cancelled"].includes(tk.status)
    );
    const interval = setInterval(refresh, hasActive ? 5000 : 30000);
    return () => clearInterval(interval);
  }, [refresh, tasks.length > 0 && tasks.some((tk) => !["done", "failed", "cancelled"].includes(tk.status))]);

  useEffect(() => {
    if (activeTaskId) {
      const timer = setTimeout(refresh, 2000);
      return () => clearTimeout(timer);
    }
  }, [activeTaskId, refresh]);

  const handleRetry = useCallback(
    async (taskId: string, e: React.MouseEvent) => {
      e.stopPropagation();
      setRetrying(taskId);
      try {
        const { task_id } = await api.retryTask(taskId);
        onRetry(task_id);
        refresh();
      } catch (err) {
        alert(t.retryFailed((err as Error).message));
      } finally {
        setRetrying(null);
      }
    },
    [onRetry, refresh, t]
  );

  const handleStop = useCallback(
    async (taskId: string, e: React.MouseEvent) => {
      e.stopPropagation();
      setStopping(taskId);
      try {
        await api.cancelTask(taskId);
        refresh();
      } catch {
        // ignore — task may have already finished
      } finally {
        setStopping(null);
      }
    },
    [refresh]
  );

  if (tasks.length === 0) return null;

  return (
    <section>
      <h2
        style={{
          fontSize: 18,
          fontWeight: 600,
          margin: "0 0 12px",
          cursor: "pointer",
          userSelect: "none",
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
        onClick={() => setCollapsed((c) => !c)}
      >
        <span
          style={{
            display: "inline-block",
            transform: collapsed ? "rotate(-90deg)" : "rotate(0deg)",
            transition: "transform 0.2s",
          }}
        >
          ▼
        </span>
        {t.taskHistory}
        <span style={{ fontSize: 13, color: "#888", fontWeight: 400 }}>
          ({tasks.length})
        </span>
      </h2>

      {!collapsed && (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 8,
            maxHeight: 320,
            overflowY: "auto",
          }}
        >
          {tasks.map((tk) => {
            const info = STATUS_LABELS[tk.status] ?? {
              label: tk.status,
              color: "#888",
            };
            const isActive = tk.task_id === activeTaskId;
            const isDone = tk.status === "done";
            const canRetry = tk.status === "failed" || tk.status === "cancelled";
            const isRunning = !["done", "failed", "cancelled"].includes(tk.status);

            return (
              <div
                key={tk.task_id}
                onClick={() => isDone && onSelect(tk.task_id)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  padding: "10px 14px",
                  background: isActive
                    ? "rgba(108,92,231,0.15)"
                    : "rgba(255,255,255,0.04)",
                  border: isActive
                    ? "1px solid rgba(108,92,231,0.4)"
                    : "1px solid rgba(255,255,255,0.08)",
                  borderRadius: 8,
                  cursor: isDone ? "pointer" : "default",
                  transition: "background 0.15s",
                }}
              >
                <span
                  style={{
                    fontSize: 12,
                    fontWeight: 600,
                    padding: "2px 8px",
                    borderRadius: 4,
                    background: info.color + "22",
                    color: info.color,
                    whiteSpace: "nowrap",
                  }}
                >
                  {info.label}
                </span>

                <div style={{ flex: 1, minWidth: 0 }}>
                  <div
                    style={{
                      fontSize: 13,
                      color: "#ccc",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {t.taskImages(tk.image_count)}
                    {tk.config
                      ? ` / ${tk.config.quality} / ${tk.config.aspect_ratio}`
                      : ""}
                  </div>
                  {tk.error_message && (
                    <div
                      style={{
                        fontSize: 11,
                        color: "#e74c3c",
                        marginTop: 2,
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                      title={tk.error_message}
                    >
                      {tk.error_message}
                    </div>
                  )}
                </div>

                {tk.duration_seconds != null && (
                  <span style={{ fontSize: 12, color: "#888" }}>
                    {formatDuration(tk.duration_seconds)}
                  </span>
                )}

                <span style={{ fontSize: 12, color: "#666", whiteSpace: "nowrap" }}>
                  {formatDate(tk.created_at)}
                </span>

                {isRunning && (
                  <button
                    onClick={(e) => handleStop(tk.task_id, e)}
                    disabled={stopping === tk.task_id}
                    style={{
                      fontSize: 12,
                      fontWeight: 600,
                      padding: "4px 10px",
                      borderRadius: 4,
                      border: "1px solid #e74c3c",
                      background: "transparent",
                      color: "#e74c3c",
                      cursor: stopping === tk.task_id ? "wait" : "pointer",
                      whiteSpace: "nowrap",
                      opacity: stopping === tk.task_id ? 0.5 : 1,
                    }}
                  >
                    {stopping === tk.task_id ? "..." : t.btnStop}
                  </button>
                )}

                {canRetry && (
                  <button
                    onClick={(e) => handleRetry(tk.task_id, e)}
                    disabled={retrying === tk.task_id}
                    style={{
                      fontSize: 12,
                      fontWeight: 600,
                      padding: "4px 10px",
                      borderRadius: 4,
                      border: "1px solid #f39c12",
                      background: "transparent",
                      color: "#f39c12",
                      cursor: retrying === tk.task_id ? "wait" : "pointer",
                      whiteSpace: "nowrap",
                      opacity: retrying === tk.task_id ? 0.5 : 1,
                    }}
                  >
                    {retrying === tk.task_id ? "..." : t.btnRetry}
                  </button>
                )}

                {isDone && (
                  <span style={{ fontSize: 16, color: "#6c5ce7" }}>▶</span>
                )}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
