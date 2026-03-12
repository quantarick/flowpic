import type { AspectRatio, ProjectConfig, Quality } from "../types";

interface Props {
  config: ProjectConfig;
  onChange: (updates: Partial<ProjectConfig>) => void;
  disabled?: boolean;
}

const ASPECT_RATIOS: { value: AspectRatio; label: string }[] = [
  { value: "16:9", label: "16:9 (Landscape)" },
  { value: "21:9", label: "21:9 (Ultrawide)" },
  { value: "9:16", label: "9:16 (Portrait)" },
  { value: "1:1", label: "1:1 (Square)" },
  { value: "4:3", label: "4:3 (Classic)" },
];

const QUALITIES: { value: Quality; label: string }[] = [
  { value: "720p", label: "720p (SD)" },
  { value: "1080p", label: "1080p (HD)" },
  { value: "2k", label: "2K (QHD)" },
  { value: "4k", label: "4K (UHD)" },
];

export function ConfigPanel({ config, onChange, disabled }: Props) {
  return (
    <div
      style={{
        display: "flex",
        gap: 16,
        flexWrap: "wrap",
        alignItems: "center",
      }}
    >
      <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
        Aspect Ratio:
        <select
          value={config.aspect_ratio}
          onChange={(e) =>
            onChange({ aspect_ratio: e.target.value as AspectRatio })
          }
          disabled={disabled}
          style={selectStyle}
        >
          {ASPECT_RATIOS.map((ar) => (
            <option key={ar.value} value={ar.value}>
              {ar.label}
            </option>
          ))}
        </select>
      </label>

      <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
        Quality:
        <select
          value={config.quality}
          onChange={(e) => onChange({ quality: e.target.value as Quality })}
          disabled={disabled}
          style={selectStyle}
        >
          {QUALITIES.map((q) => (
            <option key={q.value} value={q.value}>
              {q.label}
            </option>
          ))}
        </select>
      </label>

      <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
        FPS:
        <input
          type="number"
          min={15}
          max={60}
          value={config.fps}
          onChange={(e) => onChange({ fps: Number(e.target.value) })}
          disabled={disabled}
          style={{ ...selectStyle, width: 60 }}
        />
      </label>
    </div>
  );
}

const selectStyle: React.CSSProperties = {
  padding: "6px 10px",
  borderRadius: 6,
  border: "1px solid #555",
  background: "#2a2a2a",
  color: "#eee",
  fontSize: 14,
};
