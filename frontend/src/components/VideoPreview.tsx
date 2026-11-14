import { useI18n } from "../i18n";

interface Props {
  videoUrl: string | null;
  taskId: string | null;
}

export function VideoPreview({ videoUrl, taskId }: Props) {
  const { t } = useI18n();

  if (!videoUrl) return null;

  return (
    <div style={{ width: "100%", maxWidth: 800, margin: "0 auto" }}>
      <video
        src={videoUrl}
        controls
        autoPlay
        style={{
          // Bounded both ways, so a tall phone video fits on screen
          // instead of running far below the fold.
          display: "block",
          maxWidth: "100%",
          maxHeight: "80vh",
          margin: "0 auto",
          borderRadius: 12,
          background: "#000",
        }}
      />
      <div style={{ textAlign: "center", marginTop: 12 }}>
        <a
          href={videoUrl}
          download={`flowpic_${taskId}.mp4`}
          style={{
            display: "inline-block",
            padding: "10px 24px",
            background: "#6c5ce7",
            color: "#fff",
            borderRadius: 8,
            textDecoration: "none",
            fontWeight: 500,
          }}
        >
          {t.btnDownload}
        </a>
      </div>
    </div>
  );
}
