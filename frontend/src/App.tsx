import { useCallback, useEffect, useState } from "react";
import { ConfigPanel } from "./components/ConfigPanel";
import { CopywritingPanel } from "./components/CopywritingPanel";
import { CropPreview } from "./components/CropPreview";
import { ImageUploader } from "./components/ImageUploader";
import { LangSwitch } from "./components/LangSwitch";
import { MusicUploader } from "./components/MusicUploader";
import { ProgressBar } from "./components/ProgressBar";
import { TaskHistory } from "./components/TaskHistory";
import { VideoPreview } from "./components/VideoPreview";
import { useI18n } from "./i18n";
import { useProject } from "./hooks/useProject";
import { useWebSocket } from "./hooks/useWebSocket";
import { listCrops } from "./api/client";

type Tab = "crops" | "video" | "publish";

export default function App() {
  const { t } = useI18n();
  const project = useProject();
  const { progress, cancel } = useWebSocket(project.taskId);
  const [tab, setTab] = useState<Tab>("crops");
  const [showCrops, setShowCrops] = useState(false);
  const [cropTaskRunning, setCropTaskRunning] = useState(false);
  const [videoTaskRunning, setVideoTaskRunning] = useState(false);
  const [crops, setCrops] = useState<string[]>([]);

  // Load crops from backend (on mount + after crop task completes)
  const projectId = project.projectId;
  useEffect(() => {
    if (!projectId) return;
    listCrops(projectId)
      .then((res) => setCrops(res.crops))
      .catch(() => {});
  }, [projectId, showCrops]);

  // Track which tab started the current task
  const taskId = project.taskId;
  const progressStatus = progress?.status;
  const setDone = project.setDone;

  useEffect(() => {
    if (progressStatus === "done" && taskId) {
      if (cropTaskRunning) {
        setShowCrops(true);
        setCropTaskRunning(false);
      } else {
        setDone(taskId);
        setVideoTaskRunning(false);
      }
    }
    if (progressStatus === "failed" || progressStatus === "cancelled") {
      setCropTaskRunning(false);
      setVideoTaskRunning(false);
    }
  }, [progressStatus, taskId, setDone, cropTaskRunning]);

  const handleSelectTask = useCallback(
    (taskId: string) => {
      project.setDone(taskId);
    },
    [project.setDone]
  );

  const handleRetry = useCallback(
    (newTaskId: string) => {
      project.setTaskId(newTaskId);
      setVideoTaskRunning(true);
    },
    [project.setTaskId]
  );

  const isGenerating =
    project.taskId !== null &&
    progress !== null &&
    progress.status !== "done" &&
    progress.status !== "failed" &&
    progress.status !== "cancelled";

  const canCropPreview =
    project.images.length >= 2 && !isGenerating && !project.loading;

  const canGenerate =
    project.images.length >= 2 &&
    project.music !== null &&
    !isGenerating &&
    !project.loading;

  return (
    <div
      style={{
        maxWidth: 900,
        margin: "0 auto",
        padding: "32px 24px",
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        color: "#eee",
        minHeight: "100vh",
      }}
    >
      <header style={{ textAlign: "center", marginBottom: 24, position: "relative" }}>
        <LangSwitch />
        <h1 style={{ fontSize: 36, fontWeight: 700, margin: "0 0 8px" }}>
          {t.appTitle}
        </h1>
        <p style={{ color: "#888", margin: 0 }}>{t.appTagline}</p>
      </header>

      {/* Tab bar */}
      <div style={{ display: "flex", gap: 0, marginBottom: 24, borderBottom: "1px solid #333" }}>
        {(["crops", "video", "publish"] as Tab[]).map((t2) => {
          const active = tab === t2;
          const otherTabBusy =
            (t2 === "crops" && videoTaskRunning && isGenerating) ||
            (t2 === "video" && cropTaskRunning && isGenerating);
          const label =
            t2 === "crops" ? t.tabCrops : t2 === "video" ? t.tabVideo : t.tabPublish;
          return (
            <button
              key={t2}
              onClick={() => setTab(t2)}
              style={{
                padding: "10px 24px",
                fontSize: 15,
                fontWeight: active ? 600 : 400,
                background: "transparent",
                border: "none",
                borderBottom: active ? "2px solid #6c5ce7" : "2px solid transparent",
                color: active ? "#eee" : "#888",
                cursor: "pointer",
                position: "relative",
                transition: "all 0.15s",
              }}
            >
              {label}
              {otherTabBusy && (
                <span
                  style={{
                    display: "inline-block",
                    width: 6,
                    height: 6,
                    borderRadius: "50%",
                    background: "#6c5ce7",
                    marginLeft: 6,
                    animation: "pulse 1.2s infinite",
                  }}
                />
              )}
            </button>
          );
        })}
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
        {/* Shared: Image uploader (always visible) */}
        <section>
          <h2 style={sectionTitle}>{t.sectionImages}</h2>
          <ImageUploader
            images={project.images}
            onUpload={project.uploadImages}
            disabled={isGenerating}
          />
        </section>

        {/* ========== CROPS TAB ========== */}
        {tab === "crops" && (
          <>
            <section>
              <ConfigPanel
                config={project.config}
                onChange={project.updateConfig}
                disabled={isGenerating}
                fields={["aspect_ratio", "quality", "vision_model"]}
              />
            </section>

            <section style={{ textAlign: "center", padding: "8px 0" }}>
              <button
                onClick={() => {
                  setCropTaskRunning(true);
                  setShowCrops(false);
                  project.cropPreview();
                }}
                disabled={!canCropPreview}
                style={{
                  padding: "12px 24px",
                  fontSize: 16,
                  fontWeight: 600,
                  borderRadius: 8,
                  border: canCropPreview ? "1px solid #6c5ce7" : "1px solid #444",
                  background: canCropPreview ? "#6c5ce7" : "transparent",
                  color: canCropPreview ? "#fff" : "#666",
                  cursor: canCropPreview ? "pointer" : "not-allowed",
                  transition: "all 0.2s",
                }}
              >
                {project.loading ? t.btnStarting : t.btnCropPreview}
              </button>
            </section>

            {cropTaskRunning && project.taskId && (
              <section>
                <ProgressBar progress={progress} onCancel={cancel} />
              </section>
            )}

            {showCrops && project.projectId && (
              <section>
                <h2 style={sectionTitle}>{t.cropPreviewTitle}</h2>
                <CropPreview projectId={project.projectId} />
              </section>
            )}

            <TaskHistory
              activeTaskId={cropTaskRunning ? project.taskId : null}
              taskType="crop_preview"
              pipelineStages={["captioning_images", "reviewing_crops", "done"]}
              onSelect={() => setShowCrops(true)}
              onRetry={(newTaskId) => {
                project.setTaskId(newTaskId);
                setCropTaskRunning(true);
                setShowCrops(false);
              }}
            />
          </>
        )}

        {/* ========== VIDEO TAB ========== */}
        {tab === "video" && (
          <>
            <section>
              <h2 style={sectionTitle}>{t.sectionMusic}</h2>
              <MusicUploader
                music={project.music}
                musicDuration={project.musicDuration}
                onUpload={project.uploadMusic}
                disabled={isGenerating}
              />
            </section>

            <section>
              <ConfigPanel
                config={project.config}
                onChange={project.updateConfig}
                disabled={isGenerating}
                fields={["fps"]}
              />
            </section>

            {!showCrops && project.images.length >= 2 && (
              <p style={{ color: "#888", fontSize: 13, textAlign: "center", margin: 0 }}>
                {t.hintPreviewCrops}
              </p>
            )}

            <section style={{ textAlign: "center", padding: "8px 0" }}>
              <button
                onClick={() => {
                  setVideoTaskRunning(true);
                  project.generate();
                }}
                disabled={!canGenerate}
                style={{
                  padding: "12px 32px",
                  fontSize: 18,
                  fontWeight: 600,
                  borderRadius: 8,
                  border: "none",
                  background: canGenerate ? "#6c5ce7" : "#444",
                  color: "#fff",
                  cursor: canGenerate ? "pointer" : "not-allowed",
                  transition: "background 0.2s",
                }}
              >
                {project.loading ? t.btnStarting : t.btnGenerate}
              </button>
            </section>

            {project.error && (
              <div
                style={{
                  padding: 12,
                  background: "rgba(231,76,60,0.15)",
                  border: "1px solid #e74c3c",
                  borderRadius: 8,
                  color: "#e74c3c",
                }}
              >
                {project.error}
              </div>
            )}

            {videoTaskRunning && project.taskId && (
              <section>
                <ProgressBar progress={progress} onCancel={cancel} />
              </section>
            )}

            {project.videoUrl && (
              <section>
                <h2 style={sectionTitle}>{t.sectionResult}</h2>
                <VideoPreview videoUrl={project.videoUrl} taskId={project.taskId} />
              </section>
            )}

            <TaskHistory
              activeTaskId={project.taskId}
              onSelect={handleSelectTask}
              onRetry={handleRetry}
              taskType="video"
            />
          </>
        )}

        {/* ========== PUBLISH TAB ========== */}
        {tab === "publish" && (
          <section>
            <CopywritingPanel
              projectId={project.projectId}
              crops={crops}
              videoUrl={project.videoUrl}
              taskId={project.taskId}
              onProjectRefresh={project.refreshProject}
            />
          </section>
        )}
      </div>

      {/* Pulse animation for tab activity dot */}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
      `}</style>
    </div>
  );
}

const sectionTitle: React.CSSProperties = {
  fontSize: 18,
  fontWeight: 600,
  margin: "0 0 12px",
};
