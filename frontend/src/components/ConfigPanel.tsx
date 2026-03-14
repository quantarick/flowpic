import { useEffect, useState } from "react";
import { useI18n } from "../i18n";
import { fetchOllamaModels } from "../api/client";
import type { AspectRatio, OllamaModel, ProjectConfig, Quality } from "../types";

interface Props {
  config: ProjectConfig;
  onChange: (updates: Partial<ProjectConfig>) => void;
  disabled?: boolean;
}

export function ConfigPanel({ config, onChange, disabled }: Props) {
  const { t } = useI18n();
  const [models, setModels] = useState<OllamaModel[]>([]);
  const [defaultModel, setDefaultModel] = useState<string>("moondream");

  useEffect(() => {
    fetchOllamaModels()
      .then((res) => {
        setModels(res.models);
        setDefaultModel(res.default);
      })
      .catch(() => {
        // Fallback: just show the default
        setModels([{ name: "moondream", size: null, parameter_size: null }]);
      });
  }, []);

  const ASPECT_RATIOS: { value: AspectRatio; label: string }[] = [
    { value: "16:9", label: t.arLandscape },
    { value: "21:9", label: t.arUltrawide },
    { value: "9:16", label: t.arPortrait },
    { value: "1:1", label: t.arSquare },
    { value: "4:3", label: t.arClassic },
  ];

  const QUALITIES: { value: Quality; label: string }[] = [
    { value: "720p", label: t.q720 },
    { value: "1080p", label: t.q1080 },
    { value: "2k", label: t.q2k },
    { value: "4k", label: t.q4k },
  ];

  const formatSize = (bytes: number | null) => {
    if (!bytes) return "";
    const gb = bytes / 1e9;
    if (gb >= 1) return ` (${gb.toFixed(1)}GB)`;
    return ` (${(bytes / 1e6).toFixed(0)}MB)`;
  };

  const selectedModel = config.vision_model ?? defaultModel;

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
        {t.cfgAspectRatio}
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
        {t.cfgQuality}
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
        {t.cfgVisionModel}
        <select
          value={selectedModel}
          onChange={(e) => {
            const val = e.target.value;
            onChange({ vision_model: val === defaultModel ? null : val });
          }}
          disabled={disabled}
          style={selectStyle}
        >
          {models.map((m) => (
            <option key={m.name} value={m.name}>
              {m.name}
              {m.parameter_size ? ` (${m.parameter_size})` : formatSize(m.size)}
              {m.name === defaultModel ? " *" : ""}
            </option>
          ))}
        </select>
      </label>

      <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
        {t.cfgFps}
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
