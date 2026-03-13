import { useCallback, useRef } from "react";
import { useI18n } from "../i18n";

interface Props {
  music: string | null;
  musicDuration: number | null;
  onUpload: (file: File) => void;
  disabled?: boolean;
}

export function MusicUploader({ music, musicDuration, onUpload, disabled }: Props) {
  const { t } = useI18n();
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      if (disabled) return;
      const file = Array.from(e.dataTransfer.files).find((f) =>
        f.type.startsWith("audio/")
      );
      if (file) onUpload(file);
    },
    [onUpload, disabled]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onUpload(file);
      e.target.value = "";
    },
    [onUpload]
  );

  const formatDuration = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  return (
    <div
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      style={{
        border: "2px dashed #666",
        borderRadius: 12,
        padding: 24,
        textAlign: "center",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.5 : 1,
        minHeight: 80,
        transition: "border-color 0.2s",
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="audio/mpeg,audio/wav,audio/flac,audio/mp4,audio/m4a"
        onChange={handleChange}
        style={{ display: "none" }}
      />
      {!music ? (
        <div>
          <p style={{ fontSize: 18, margin: "0 0 8px" }}>
            {t.musicDropHint}
          </p>
          <p style={{ color: "#888", margin: 0 }}>
            {t.musicFormatHint}
          </p>
        </div>
      ) : (
        <div>
          <p style={{ margin: "0 0 4px" }}>{music}</p>
          {musicDuration && (
            <p style={{ color: "#888", margin: 0, fontSize: 14 }}>
              {t.musicDuration} {formatDuration(musicDuration)}
            </p>
          )}
          <p style={{ color: "#888", margin: "4px 0 0", fontSize: 14 }}>
            {t.musicReplace}
          </p>
        </div>
      )}
    </div>
  );
}
