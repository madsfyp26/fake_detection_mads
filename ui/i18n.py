"""Localized UI strings for Streamlit (Home + Research pages)."""

from __future__ import annotations

SUPPORTED_LANGS = ("en", "es", "hi")


def t(key: str, lang: str) -> str:
    lang = lang if lang in SUPPORTED_LANGS else "en"
    table = STRINGS.get(lang) or STRINGS["en"]
    return table.get(key) or STRINGS["en"].get(key, key)


STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "sidebar_blurb": "Full pipeline: dataset → preprocessing → features → models → XAI.",
        "home_subheader": (
            "Multimodal deepfake screening: <strong>audio</strong>, <strong>video</strong>, and "
            "<strong>visual (face/mouth) frames</strong> with optional <strong>multilingual</strong> use — "
            "performance depends on training data and language."
        ),
        "home_recommended": (
            "**Recommended path:** Sidebar → **Inference Demo** → **Combined (AVH → NOMA)** → optional **Final Combined Report** "
            "for a one-page JSON summary."
        ),
        "home_disclaimer": (
            "This app is a **screening / research aid** — not legal evidence, medical advice, or a substitute for expert review."
        ),
        "home_quickstart_title": "**Quick start**",
        "home_q1": "In the sidebar, set **Python for AVH video** to your `avh` conda interpreter (required for video).",
        "home_q2": "Open **Inference Demo** and select **Combined (AVH → NOMA)** (default).",
        "home_q3": "Upload a short MP4 (e.g. under ~60s on a laptop) and run — then open **Final Combined Report** if you want the export.",
        "home_expander_compare": "📊 Modalities at a glance",
        "home_modality_note": (
            "**How inputs map to modalities:** **Audio** → NOMA (speech screening). "
            "**Video** → AVH-Align reads **audio + visual**: each timestep uses **image crops** of the mouth/face plus the waveform "
            "(lip–speech alignment). This app does **not** accept a standalone still photo for AVH; use **video** or **pre-extracted .npz** "
            "with `visual` + `audio` tensors. **Research chat** is text-only (any language in your question is passed through to retrieval + Gemini)."
        ),
        "home_noma_title": "🎧 NOMA",
        "home_noma_sub": "Audio · Lightweight",
        "home_noma_li1": "Input: WAV / MP3 / OGG",
        "home_noma_li2": "Targets: TTS, voice cloning, synthetic speech",
        "home_noma_li3": "Features + SVM → Real / Fake per second",
        "home_noma_li4": "Fast (seconds)",
        "home_avh_title": "🎬 AVH-Align",
        "home_avh_sub": "Video + visual frames · Audio-visual model",
        "home_avh_li1": "Input: Video (MP4 / AVI / MOV) or .npz embeddings",
        "home_avh_li2": "Visual: mouth/face **image sequence** + audio waveform",
        "home_avh_li3": "AV-HuBERT + Fusion → lip–speech mismatch score",
        "home_avh_li4": "Grad-CAM overlays = image evidence on ROI",
        "home_avh_li5": "Typ. 1–3 min on CPU",
        "home_table_header": "| | **NOMA** | **AVH-Align** |",
        "home_table_sep": "|---|---|---|",
        "home_table_r1": "| **Modality** | Audio only | Video + visual frames + audio |",
        "home_table_r2": "| **Detects** | Synthetic / cloned speech | Lip–speech mismatch |",
        "home_table_r3": "| **Method** | Hand-crafted features + SVM | AV-HuBERT + Fusion MLP |",
        "home_table_r4": "| **Output** | Per-second Real/Fake | One score (+ optional XAI images) |",
        "research_title": "## Research chat",
        "research_caption": (
            "Uses SerpAPI (organic web + Google Lens), News API, and Gemini. "
            "**Ask in any language** you like; answers follow your language when the model can. "
            "Not legal advice; third-party data and models can be incomplete or wrong."
        ),
        "research_include_combined": "Include last Combined run summary in context",
        "research_include_help": "Passes AVH / fusion fields from the latest successful Combined run to Gemini.",
        "research_env_expander": "Environment variables (API keys)",
        "research_env_md": (
            "Set `SERPAPI_API_KEY` (web + Google Lens), `NEWS_API_KEY`, `GEMINI_API_KEY` "
            "(and optionally `GEMINI_MODEL`). Copy `.env.example` in the repo root to `.env` for local dev."
        ),
        "research_show_sources": "Show raw source snippets (debug)",
        "research_chat_placeholder": "Ask about a claim, topic, or news story…",
        "research_clear": "Clear chat history",
    },
    "es": {
        "sidebar_blurb": "Flujo completo: datos → preprocesado → características → modelos → XAI.",
        "home_subheader": (
            "Detección multimodal de deepfakes: <strong>audio</strong>, <strong>vídeo</strong> y "
            "<strong>fotogramas visuales (cara/boca)</strong> con uso <strong>multilingüe</strong> opcional — "
            "el rendimiento depende de los datos de entrenamiento y del idioma."
        ),
        "home_recommended": (
            "**Ruta recomendada:** barra lateral → **Inference Demo** → **Combined (AVH → NOMA)** → opcional **Final Combined Report** "
            "para un resumen JSON en una página."
        ),
        "home_disclaimer": (
            "Esta app es una **ayuda de cribado / investigación** — no constituye prueba legal, consejo médico ni sustituto de revisión experta."
        ),
        "home_quickstart_title": "**Inicio rápido**",
        "home_q1": "En la barra lateral, configura **Python for AVH video** al intérprete conda `avh` (necesario para vídeo).",
        "home_q2": "Abre **Inference Demo** y elige **Combined (AVH → NOMA)** (por defecto).",
        "home_q3": "Sube un MP4 corto (p. ej. ~60 s en portátil) y ejecuta — luego abre **Final Combined Report** si quieres exportar.",
        "home_expander_compare": "📊 Modalidades de un vistazo",
        "home_modality_note": (
            "**Entradas y modalidades:** **Audio** → NOMA (voz). "
            "**Vídeo** → AVH-Align usa **audio + visual**: en cada instante, **recortes de imagen** de boca/cara y la forma de onda "
            "(alineación labio–voz). Esta app **no** acepta una foto fija para AVH; usa **vídeo** o **.npz** preextraído con tensores `visual` + `audio`. "
            "**Research chat** solo admite **texto** (cualquier idioma en la pregunta)."
        ),
        "home_noma_title": "🎧 NOMA",
        "home_noma_sub": "Audio · Ligero",
        "home_noma_li1": "Entrada: WAV / MP3 / OGG",
        "home_noma_li2": "Objetivos: TTS, clonación de voz, voz sintética",
        "home_noma_li3": "Características + SVM → Real / Fake por segundo",
        "home_noma_li4": "Rápido (segundos)",
        "home_avh_title": "🎬 AVH-Align",
        "home_avh_sub": "Vídeo + fotogramas · Modelo audiovisual",
        "home_avh_li1": "Entrada: vídeo (MP4 / AVI / MOV) o embeddings .npz",
        "home_avh_li2": "Visual: **secuencia de imagen** boca/cara + audio",
        "home_avh_li3": "AV-HuBERT + Fusion → puntuación de desajuste labio–voz",
        "home_avh_li4": "Grad-CAM = evidencia en imagen sobre ROI",
        "home_avh_li5": "Típ. 1–3 min en CPU",
        "home_table_header": "| | **NOMA** | **AVH-Align** |",
        "home_table_sep": "|---|---|---|",
        "home_table_r1": "| **Modalidad** | Solo audio | Vídeo + fotogramas + audio |",
        "home_table_r2": "| **Detecta** | Voz sintética / clonada | Desajuste labio–voz |",
        "home_table_r3": "| **Método** | Características + SVM | AV-HuBERT + Fusion MLP |",
        "home_table_r4": "| **Salida** | Real/Fake por segundo | Una puntuación (+ XAI opcional) |",
        "research_title": "## Chat de investigación / verificación",
        "research_caption": (
            "Usa SerpAPI (web + Google Lens), News API y Gemini. "
            "**Pregunta en el idioma que quieras**; las respuestas siguen tu idioma cuando el modelo puede. "
            "No es asesoramiento legal; los datos de terceros pueden ser incompletos o erróneos."
        ),
        "research_include_combined": "Incluir resumen del último Combined en el contexto",
        "research_include_help": "Pasa campos AVH / fusión del último Combined exitoso a Gemini.",
        "research_env_expander": "Variables de entorno (claves API)",
        "research_env_md": (
            "Define `SERPAPI_API_KEY` (web + Google Lens), `NEWS_API_KEY`, `GEMINI_API_KEY` "
            "(y opcionalmente `GEMINI_MODEL`). Copia `.env.example` a `.env` en la raíz del repo."
        ),
        "research_show_sources": "Mostrar fragmentos de fuentes (depuración)",
        "research_chat_placeholder": "Pregunta sobre una afirmación, tema o noticia…",
        "research_clear": "Borrar historial del chat",
    },
    "hi": {
        "sidebar_blurb": "पूरा पाइपलाइन: डेटासेट → प्रीप्रोसेसिंग → फ़ीचर → मॉडल → XAI।",
        "home_subheader": (
            "<strong>बहु-मोडल</strong> डीपफ़ेक स्क्रीनिंग: <strong>ऑडियो</strong>, <strong>वीडियो</strong>, और "
            "<strong>दृश्य (चेहरा/मुँह) फ़्रेम</strong> — <strong>बहुभाषी</strong> उपयोग संभव; प्रदर्शन प्रशिक्षण डेटा और भाषा पर निर्भर करता है।"
        ),
        "home_recommended": (
            "**अनुशंसित:** साइडबार → **Inference Demo** → **Combined (AVH → NOMA)** → वैकल्पिक **Final Combined Report** "
            "एक पृष्ठ JSON सारांश के लिए।"
        ),
        "home_disclaimer": (
            "यह ऐप **स्क्रीनिंग / शोध सहायक** है — कानूनी साक्ष्य, चिकित्सा सलाह या विशेषज्ञ समीक्षा का विकल्प नहीं।"
        ),
        "home_quickstart_title": "**त्वरित शुरुआत**",
        "home_q1": "साइडबार में **Python for AVH video** अपने `avh` conda इंटरप्रेटर पर सेट करें (वीडियो के लिए आवश्यक)।",
        "home_q2": "**Inference Demo** खोलें और **Combined (AVH → NOMA)** चुनें (डिफ़ॉल्ट)।",
        "home_q3": "छोटा MP4 अपलोड करें (~60s तक) और चलाएँ — फिर निर्यात के लिए **Final Combined Report** खोलें।",
        "home_expander_compare": "📊 मोडलिटी सारांश",
        "home_modality_note": (
            "**इनपुट:** **ऑडियो** → NOMA (भाषण)। "
            "**वीडियो** → AVH-Align **ऑडियो + दृश्य** लेता है: हर समय चरण पर **मुँह/चेहरे की इमेज क्रॉप** + तरंगरूप (होंठ–भाषण संरेखण)। "
            "AVH के लिए अलग से स्थिर फोटो अपलोड **नहीं**; **वीडियो** या `visual` + `audio` वाला **.npz** उपयोग करें। "
            "**Research chat** केवल **टेक्स्ट** (प्रश्न किसी भी भाषा में)।"
        ),
        "home_noma_title": "🎧 NOMA",
        "home_noma_sub": "ऑडियो · हल्का",
        "home_noma_li1": "इनपुट: WAV / MP3 / OGG",
        "home_noma_li2": "लक्ष्य: TTS, आवाज़ क्लोनिंग, संश्लिष्ट भाषण",
        "home_noma_li3": "फ़ीचर + SVM → प्रति सेकंड Real / Fake",
        "home_noma_li4": "तेज़ (सेकंड)",
        "home_avh_title": "🎬 AVH-Align",
        "home_avh_sub": "वीडियो + दृश्य फ़्रेम · ऑडियो-दृश्य मॉडल",
        "home_avh_li1": "इनपुट: वीडियो (MP4 / AVI / MOV) या .npz एम्बेडिंग",
        "home_avh_li2": "दृश्य: **मुँह/चेहरे की इमेज अनुक्रम** + ऑडियो",
        "home_avh_li3": "AV-HuBERT + Fusion → होंठ–भाषण बेमेल स्कोर",
        "home_avh_li4": "Grad-CAM ओवरले = ROI पर इमेज साक्ष्य",
        "home_avh_li5": "CPU पर लगभग 1–3 मिनट",
        "home_table_header": "| | **NOMA** | **AVH-Align** |",
        "home_table_sep": "|---|---|---|",
        "home_table_r1": "| **मोडलिटी** | केवल ऑडियो | वीडियो + दृश्य फ़्रेम + ऑडियो |",
        "home_table_r2": "| **पता लगाता है** | संश्लिष्ट / क्लोन्ड भाषण | होंठ–भाषण बेमेल |",
        "home_table_r3": "| **विधि** | फ़ीचर + SVM | AV-HuBERT + Fusion MLP |",
        "home_table_r4": "| **आउटपुट** | प्रति सेकंड Real/Fake | एक स्कोर (+ वैकल्पिक XAI) |",
        "research_title": "## अनुसंधान / तथ्य-जाँच चैट",
        "research_caption": (
            "SerpAPI (वेब + Google Lens), News API और Gemini उपयोग करता है। "
            "**किसी भी भाषा में पूछें**; जहाँ मॉडल सक्षम हो, उत्तर उसी भाषा में। "
            "कानूनी सलाह नहीं; तृतीय-पक्ष डेटा अधूरा या गलत हो सकता है।"
        ),
        "research_include_combined": "अंतिम Combined रन का सार संदर्भ में शामिल करें",
        "research_include_help": "सफल Combined रन से AVH / फ़्यूज़न फ़ील्ड Gemini को भेजता है।",
        "research_env_expander": "पर्यावरण चर (API कुंजियाँ)",
        "research_env_md": (
            "`SERPAPI_API_KEY` (वेब + Google Lens), `NEWS_API_KEY`, `GEMINI_API_KEY` सेट करें "
            "(वैकल्पिक `GEMINI_MODEL`)। लोकल विकास के लिए `.env.example` को `.env` में कॉपी करें।"
        ),
        "research_show_sources": "कच्चे स्रोत अंश दिखाएँ (डीबग)",
        "research_chat_placeholder": "दावे, विषय या समाचार के बारे में पूछें…",
        "research_clear": "चैट इतिहास साफ़ करें",
    },
}
