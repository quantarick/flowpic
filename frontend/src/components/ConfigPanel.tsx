import { useEffect, useState } from "react";
import { useI18n } from "../i18n";
import { fetchLlmModels, fetchVideoStyles } from "../api/client";
import type {
  AspectRatio,
  FitStrategy,
  FrameStyle,
  LlmModel,
  LlmModelsResponse,
  ProjectConfig,
  Quality,
  VideoStyleInfo,
} from "../types";

type ConfigField =
  | "aspect_ratio"
  | "quality"
  | "vision_model"
  | "fps"
  | "fit_strategy"
  | "video_style";

interface Props {
  config: ProjectConfig;
  onChange: (updates: Partial<ProjectConfig>) => void;
  disabled?: boolean;
  fields?: ConfigField[];
}

interface ModelOption {
  provider: string;
  configured: boolean;
  model: LlmModel;
}

// Ollama treats an untagged name as ":latest", so compare names that way.
const withTag = (name: string) => (name.includes(":") ? name : `${name}:latest`);
const sameModel = (a: string, b: string) => withTag(a) === withTag(b);

export function ConfigPanel({ config, onChange, disabled, fields }: Props) {
  const { t, lang } = useI18n();
  const [catalog, setCatalog] = useState<LlmModelsResponse | null>(null);
  const [styles, setStyles] = useState<VideoStyleInfo[]>([]);

  useEffect(() => {
    fetchLlmModels()
      .then(setCatalog)
      .catch(() => setCatalog(null));
    // The list comes from the server's style plugins, so a new style shows up with no change here.
    fetchVideoStyles()
      .then((res) => setStyles(res.styles))
      .catch(() => setStyles([]));
  }, []);

  const show = (f: ConfigField) => !fields || fields.includes(f);

  const ASPECT_RATIOS: { value: AspectRatio; label: string }[] = [
    { value: "16:9", label: t.arLandscape },
    { value: "21:9", label: t.arUltrawide },
    { value: "9:16", label: t.arPortrait },
    { value: "9:19.5", label: t.arPhone },
    { value: "3:4", label: t.arPortraitPhoto },
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

  // One flat list of choices across providers; the select's value is an index into it.
  const options: ModelOption[] = (catalog?.providers ?? []).flatMap((p) =>
    p.models.map((model) => ({ provider: p.name, configured: p.configured, model }))
  );
  const serverDefault = catalog?.default ?? null;
  const isDefault = (o: ModelOption) =>
    serverDefault !== null &&
    o.provider === serverDefault.provider &&
    sameModel(o.model.name, serverDefault.model);

  // A project saved before providers existed has a model and no provider: that model is Ollama's.
  const selectedProvider = config.vision_model
    ? config.vision_provider ?? "ollama"
    : serverDefault?.provider ?? null;
  const selectedModel = config.vision_model ?? serverDefault?.model ?? null;
  const selectedIndex = options.findIndex(
    (o) =>
      o.provider === selectedProvider &&
      selectedModel !== null &&
      sameModel(o.model.name, selectedModel)
  );

  return (
    <div
      style={{
        display: "flex",
        gap: 16,
        flexWrap: "wrap",
        alignItems: "center",
      }}
    >
      {show("aspect_ratio") && (
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
      )}

      {show("quality") && (
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
      )}

      {show("vision_model") && (
        <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {t.cfgVisionModel}
          <select
            value={selectedIndex >= 0 ? String(selectedIndex) : ""}
            onChange={(e) => {
              const picked = options[Number(e.target.value)];
              if (!picked) return;
              onChange(
                isDefault(picked)
                  ? { vision_provider: null, vision_model: null }
                  : { vision_provider: picked.provider, vision_model: picked.model.name }
              );
            }}
            disabled={disabled}
            style={selectStyle}
          >
            {selectedIndex < 0 && (
              <option value="">
                {selectedModel
                  ? `${selectedProvider ? `${selectedProvider} / ` : ""}${selectedModel}`
                  : "-"}
              </option>
            )}
            {(catalog?.providers ?? []).map((p) => (
              <optgroup
                key={p.name}
                label={p.configured ? p.name : `${p.name} (${t.cfgNoApiKey})`}
              >
                {options.map((o, i) =>
                  o.provider !== p.name ? null : (
                    <option key={`${o.provider}/${o.model.name}`} value={String(i)}>
                      {o.model.name}
                      {o.model.parameter_size
                        ? ` (${o.model.parameter_size})`
                        : formatSize(o.model.size)}
                      {isDefault(o) ? " *" : ""}
                    </option>
                  )
                )}
              </optgroup>
            ))}
          </select>
        </label>
      )}

      {show("video_style") && (
        <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {t.cfgVideoStyle}
          <select
            value={config.video_style}
            onChange={(e) => onChange({ video_style: e.target.value })}
            disabled={disabled}
            style={selectStyle}
            title={
              styles.find((s) => s.name === config.video_style)?.description[lang] ?? undefined
            }
          >
            {!styles.some((s) => s.name === config.video_style) && (
              <option value={config.video_style}>{config.video_style}</option>
            )}
            {styles.map((s) => (
              <option key={s.name} value={s.name} title={s.description[lang] ?? s.description.en}>
                {s.label[lang] ?? s.label.en ?? s.name}
              </option>
            ))}
          </select>
        </label>
      )}

      {show("fps") && (
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
      )}

      {show("fit_strategy") && (
        <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {t.cfgFitStrategy}
          <select
            value={config.fit_strategy}
            onChange={(e) =>
              onChange({ fit_strategy: e.target.value as FitStrategy })
            }
            disabled={disabled}
            style={selectStyle}
          >
            <option value="crop">{t.fitCrop}</option>
            <option value="frame">{t.fitFrame}</option>
          </select>
        </label>
      )}

      {show("fit_strategy") && config.fit_strategy === "frame" && (
        <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {t.cfgFrameStyle}
          <select
            value={config.frame_style}
            onChange={(e) =>
              onChange({ frame_style: e.target.value as FrameStyle })
            }
            disabled={disabled}
            style={selectStyle}
          >
            <option value="film">{t.frameFilm}</option>
            <option value="clean">{t.frameClean}</option>
            <option value="polaroid">{t.framePolaroid}</option>
            <option value="shadow">{t.frameShadow}</option>
          </select>
        </label>
      )}
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
