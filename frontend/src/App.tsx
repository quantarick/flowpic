import { useEffect } from "react";
import { ConfigPanel } from "./components/ConfigPanel";
import { GenerateButton } from "./components/GenerateButton";
import { ImageUploader } from "./components/ImageUploader";
import { MusicUploader } from "./components/MusicUploader";
import { ProgressBar } from "./components/ProgressBar";
import { VideoPreview } from "./components/VideoPreview";
import { useProject } from "./hooks/useProject";
import { useWebSocket } from "./hooks/useWebSocket";
export default function App() {
  const project = useProject();
  const { progress, cancel } = useWebSocket(project.taskId);

  // Update video URL when done
  const taskId = project.taskId;
  const progressStatus = progress?.status;
  const setDone = project.setDone;
  useEffect(() => {
    if (progressStatus === "done" && taskId) {
      setDone(taskId);
    }
  }, [progressStatus, taskId, setDone]);

  const isGenerating =
    project.taskId !== null &&
    progress !== null &&
    progress.status !== "done" &&
    progress.status !== "failed" &&
    progress.status !== "cancelled";

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
      <header style={{ textAlign: "center", marginBottom: 40 }}>
        <h1 style={{ fontSize: 36, fontWeight: 700, margin: "0 0 8px" }}>
          FlowPic
        </h1>
        <p style={{ color: "#888", margin: 0 }}>
          Music-driven photo slideshow generator
        </p>
      </header>

      <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
        <section>
          <h2 style={sectionTitle}>1. Upload Images</h2>
          <ImageUploader
            images={project.images}
            onUpload={project.uploadImages}
            disabled={isGenerating}
          />
        </section>

        <section>
          <h2 style={sectionTitle}>2. Upload Music</h2>
          <MusicUploader
            music={project.music}
            musicDuration={project.musicDuration}
            onUpload={project.uploadMusic}
            disabled={isGenerating}
          />
        </section>

        <section>
          <h2 style={sectionTitle}>3. Configure</h2>
          <ConfigPanel
            config={project.config}
            onChange={project.updateConfig}
            disabled={isGenerating}
          />
        </section>

        <section style={{ textAlign: "center", padding: "8px 0" }}>
          <GenerateButton
            canGenerate={canGenerate}
            loading={project.loading}
            onClick={project.generate}
          />
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

        {project.taskId && (
          <section>
            <ProgressBar progress={progress} onCancel={cancel} />
          </section>
        )}

        {project.videoUrl && (
          <section>
            <h2 style={sectionTitle}>Result</h2>
            <VideoPreview videoUrl={project.videoUrl} taskId={project.taskId} />
          </section>
        )}
      </div>
    </div>
  );
}

const sectionTitle: React.CSSProperties = {
  fontSize: 18,
  fontWeight: 600,
  margin: "0 0 12px",
};
