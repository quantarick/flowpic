import { createContext, useCallback, useContext, useState } from "react";
import en, { type Translations } from "./en";
import zh from "./zh";

export type Lang = "en" | "zh";

const translations: Record<Lang, Translations> = { en, zh };

interface I18nContext {
  lang: Lang;
  t: Translations;
  setLang: (lang: Lang) => void;
}

const Ctx = createContext<I18nContext>({
  lang: "en",
  t: en,
  setLang: () => {},
});

function detectLang(): Lang {
  const saved = localStorage.getItem("flowpic_lang");
  if (saved === "zh" || saved === "en") return saved;
  const nav = navigator.language.toLowerCase();
  if (nav.startsWith("zh")) return "zh";
  return "en";
}

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLangState] = useState<Lang>(detectLang);

  const setLang = useCallback((l: Lang) => {
    setLangState(l);
    localStorage.setItem("flowpic_lang", l);
  }, []);

  return (
    <Ctx.Provider value={{ lang, t: translations[lang], setLang }}>
      {children}
    </Ctx.Provider>
  );
}

export function useI18n() {
  return useContext(Ctx);
}
