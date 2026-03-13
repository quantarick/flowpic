import { useI18n, type Lang } from "../i18n";

const LANGS: { value: Lang; label: string }[] = [
  { value: "en", label: "EN" },
  { value: "zh", label: "中文" },
];

export function LangSwitch() {
  const { lang, setLang } = useI18n();

  return (
    <div style={{ position: "absolute", top: 0, right: 0, display: "flex", gap: 4 }}>
      {LANGS.map((l) => (
        <button
          key={l.value}
          onClick={() => setLang(l.value)}
          style={{
            padding: "4px 10px",
            fontSize: 12,
            fontWeight: lang === l.value ? 700 : 400,
            borderRadius: 4,
            border: lang === l.value ? "1px solid #6c5ce7" : "1px solid #555",
            background: lang === l.value ? "rgba(108,92,231,0.2)" : "transparent",
            color: lang === l.value ? "#6c5ce7" : "#888",
            cursor: "pointer",
          }}
        >
          {l.label}
        </button>
      ))}
    </div>
  );
}
