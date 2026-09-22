import streamlit as st
from textblob import TextBlob
from deep_translator import GoogleTranslator, MyMemoryTranslator
from langdetect import detect
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import re


# Argos guarda los modelos dentro del proyecto, no en el perfil del usuario.
ARGOS_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("XDG_DATA_HOME", str(ARGOS_ROOT / ".argos-data"))
os.environ.setdefault("XDG_CONFIG_HOME", str(ARGOS_ROOT / ".argos-config"))
os.environ.setdefault("XDG_CACHE_HOME", str(ARGOS_ROOT / ".argos-cache"))

from argostranslate import translate as argos_translate

st.set_page_config(
    page_title="SentimentAI | Análisis de Sentimiento",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>

    /* Fondo general */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #111827 50%, #020617 100%);
        color: #f8fafc;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #020617;
        border-right: 1px solid #1e293b;
    }

    /* Títulos */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* Tarjetas */
    .card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid #1e293b;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }

    .sentiment-positive {
        color: #4ade80;
        font-size: 2rem;
        font-weight: 800;
    }

    .sentiment-negative {
        color: #f87171;
        font-size: 2rem;
        font-weight: 800;
    }

    .sentiment-neutral {
        color: #facc15;
        font-size: 2rem;
        font-weight: 800;
    }

    .small-text {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    /* Métricas */
    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #1e293b;
        padding: 15px;
        border-radius: 14px;
    }

    /* Botones */
    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        height: 45px;
    }

    /* Textarea */
    textarea {
        border-radius: 14px !important;
    }

</style>
""", unsafe_allow_html=True)


if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None



SPANISH_POSITIVE_WORDS = {
    "amo", "amamos", "bueno", "buena", "buenisimo", "excelente",
    "encanta", "encanto", "feliz", "fantastico", "genial", "gusta",
    "increible", "maravilloso", "mejor", "perfecto", "positivo",
    "recomiendo", "satisfecho", "satisfecha"
}

SPANISH_NEGATIVE_WORDS = {
    "aburrido", "decepcion", "decepcionado", "decepcionante", "dolor",
    "horrible", "malo", "mala", "molesto", "negativo", "odio",
    "maldad", "peor", "pesimo", "problema", "terrible", "triste"
}

ENGLISH_POSITIVE_WORDS = {
    "amazing", "awesome", "best", "excellent", "fantastic", "fascinate",
    "fascinates", "fascinating", "good", "great", "happy", "like", "love",
    "loved", "lovely", "perfect", "recommend", "satisfied", "wonderful"
}

ENGLISH_NEGATIVE_WORDS = {
    "awful", "bad", "boring", "disappointing", "hate", "horrible", "mad",
    "negative", "problem", "sad", "terrible", "upset", "worst"
}


def detect_language(text):
    """Devuelve el idioma detectado sin interrumpir el análisis."""
    try:
        return detect(text)
    except Exception:
        return "unknown"


def analyze_spanish_sentiment(text):
    """Respaldo local básico para textos en español sin conexión."""
    words = re.findall(r"[a-záéíóúüñ]+", text.lower())
    positive = sum(word in SPANISH_POSITIVE_WORDS for word in words)
    negative = sum(word in SPANISH_NEGATIVE_WORDS for word in words)
    matches = positive + negative

    if not matches:
        return 0.0, 0.0

    polarity = (positive - negative) / matches
    subjectivity = min(1.0, matches / max(1, len(words) * 0.25))
    return polarity, subjectivity


def analyze_english_sentiment(text):
    """Respaldo léxico para expresiones que TextBlob deja en neutral."""
    words = re.findall(r"[a-z]+", text.lower())
    positive = sum(word in ENGLISH_POSITIVE_WORDS for word in words)
    negative = sum(word in ENGLISH_NEGATIVE_WORDS for word in words)
    matches = positive + negative

    if not matches:
        return 0.0, 0.0

    polarity = (positive - negative) / matches
    subjectivity = min(1.0, matches / max(1, len(words) * 0.25))
    return polarity, subjectivity


def translate_text(text):
    """Traduce español a inglés, priorizando el modelo local de Argos."""
    try:
        translated = argos_translate.translate(text, "es", "en")
        if translated and translated.strip():
            return translated.strip(), None
    except Exception as error:
        argos_error = f"ArgosTranslate: {error}"
    else:
        argos_error = "ArgosTranslate: no devolvió texto"

    translators = (
        (GoogleTranslator, {"source": "es", "target": "en"}),
        (MyMemoryTranslator, {"source": "spanish", "target": "english"}),
    )
    errors = [argos_error]

    for translator_class, options in translators:
        try:
            translated = translator_class(**options).translate(text)
            if translated and translated.strip():
                return translated.strip(), None
            errors.append(f"{translator_class.__name__}: no devolvió texto")
        except Exception as error:
            errors.append(f"{translator_class.__name__}: {error}")

    return text, " | ".join(errors)


def classify_sentiment(polarity):
    """
    Clasifica el sentimiento según la polaridad.
    """

    if polarity > 0.1:
        return {
            "label": "Positivo",
            "emoji": "😊",
            "color": "#4ade80",
            "description": "El texto presenta una tendencia emocional positiva."
        }

    elif polarity < -0.1:
        return {
            "label": "Negativo",
            "emoji": "😞",
            "color": "#f87171",
            "description": "El texto presenta una tendencia emocional negativa."
        }

    return {
        "label": "Neutral",
        "emoji": "😐",
        "color": "#facc15",
        "description": "El texto presenta una tendencia emocional neutral."
    }


def analyze_text(text, use_translation=False):

    if not text or not text.strip():
        return None

    original_text = text.strip()
    analyzed_text = original_text
    translation_error = None
    language = detect_language(original_text)

    # Traducción opcional
    if use_translation:
        analyzed_text, translation_error = translate_text(original_text)

    # Análisis
    translated = analyzed_text != original_text

    # Si la traducción solicitada falla, el texto sigue siendo español:
    # no dependemos de la detección automática para aplicar el respaldo local.
    if not translated and (language == "es" or (use_translation and translation_error)):
        polarity, subjectivity = analyze_spanish_sentiment(original_text)
    else:
        blob = TextBlob(analyzed_text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        if polarity == 0.0:
            fallback_polarity, fallback_subjectivity = analyze_english_sentiment(
                analyzed_text
            )
            if fallback_polarity != 0.0:
                polarity, subjectivity = fallback_polarity, fallback_subjectivity

    sentiment = classify_sentiment(polarity)

    words = analyzed_text.split()

    result = {
        "original_text": original_text,
        "analyzed_text": analyzed_text,
        "sentiment": sentiment["label"],
        "emoji": sentiment["emoji"],
        "color": sentiment["color"],
        "description": sentiment["description"],
        "polarity": polarity,
        "subjectivity": subjectivity,
        "word_count": len(words),
        "character_count": len(original_text),
        "translated": translated,
        "translation_error": translation_error,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }

    return result


def create_polarity_chart(polarity):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=polarity,
            number={
                "valueformat": ".3f"
            },
            title={
                "text": "Polaridad"
            },
            gauge={
                "axis": {
                    "range": [-1, 1]
                },
                "bar": {
                    "color": "#38bdf8"
                },
                "steps": [
                    {
                        "range": [-1, -0.1],
                        "color": "#7f1d1d"
                    },
                    {
                        "range": [-0.1, 0.1],
                        "color": "#713f12"
                    },
                    {
                        "range": [0.1, 1],
                        "color": "#166534"
                    }
                ],
            }
        )
    )

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    return fig


def create_subjectivity_chart(subjectivity):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=subjectivity * 100,
            number={
                "suffix": "%",
                "valueformat": ".1f"
            },
            title={
                "text": "Subjetividad"
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "bar": {
                    "color": "#818cf8"
                }
            }
        )
    )

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    return fig


with st.sidebar:

    st.markdown("## 🧠 SentimentAI")

    st.markdown(
        "### Análisis inteligente de texto"
    )

    st.divider()

    st.markdown("### ⚙️ Configuración")

    use_translation = st.toggle(
        "🌐 Traducir español → inglés",
        value=False
    )

    st.caption(
        "Habilita esta opcion para traducir el texto de español a inglés antes de analizarlo si ya esta en ingles, omitelo. "
        "TextBlob funciona principalmente con modelos "
        "basados en inglés. La traducción puede mejorar "
        "los resultados en textos escritos en español."
    )

    st.divider()

    st.markdown("### 📊 Sesión")

    total = len(st.session_state.history)

    positive = sum(
        1 for item in st.session_state.history
        if item["sentiment"] == "Positivo"
    )

    negative = sum(
        1 for item in st.session_state.history
        if item["sentiment"] == "Negativo"
    )

    neutral = sum(
        1 for item in st.session_state.history
        if item["sentiment"] == "Neutral"
    )

    st.metric("Análisis realizados", total)

    c1, c2, c3 = st.columns(3)

    c1.metric("😊", positive)
    c2.metric("😐", neutral)
    c3.metric("😞", negative)

    st.divider()

    if st.button(
        "🗑️ Limpiar historial",
        use_container_width=True
    ):
        st.session_state.history = []
        st.session_state.last_result = None
        st.rerun()


st.markdown(
    '<div class="main-title">🧠 SentimentAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Sistema de análisis de sentimiento basado en procesamiento '
    'de lenguaje natural.'
    '</div>'
    '<div class="subtitle">'
        'Presentado por Jefferson bonilla y Ricardo Martinez. REG-SAN MIGUEL'
        '</div>',
    unsafe_allow_html=True
)


def clear_input_text():
    """Vacía el contenido del área de texto antes de volver a dibujarla."""
    st.session_state.user_text = ""


st.markdown("## ✍️ Analizar texto")

with st.container(border=True):

    user_text = st.text_area(
        "Introduce el texto que deseas analizar",
        placeholder=(
            "Ejemplo:\n\n"
            "The car is amazing and I really love its performance."
        ),
        height=160,
        label_visibility="visible",
        key="user_text"
    )

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        analyze_button = st.button(
            "🚀 Analizar sentimiento",
            type="primary",
            use_container_width=True
        )

    with col2:
        st.button(
            "🧹 Limpiar texto",
            use_container_width=True,
            on_click=clear_input_text
        )

    with col3:
        st.caption(
            f"Caracteres: {len(user_text)}"
        )

if analyze_button:

    if not user_text.strip():

        st.warning(
            "⚠️ Introduce algún texto antes de analizar."
        )

    else:

        with st.spinner("🧠 Procesando lenguaje natural..."):

            result = analyze_text(
                user_text,
                use_translation
            )

        if result:
            st.session_state.last_result = result
            st.session_state.history.append(result)


result = st.session_state.last_result

if result:

    st.markdown("---")

    st.markdown("## 📊 Resultado del análisis")

    st.markdown("### Resultado")

    col_emoji, col_info = st.columns([1, 5])

    with col_emoji:
        st.markdown(
            f"# {result['emoji']}"
        )

    sentiment = result["sentiment"]

    if sentiment == "Positivo":
        col_info.success(
            f"😊 {sentiment}",
            icon="✅"
        )

    elif sentiment == "Negativo":
        col_info.error(
            f"😞 {sentiment}",
            icon="❌"
        )

    else:
        col_info.warning(
            f"😐 {sentiment}",
            icon="➖"
        )

    col_info.caption(result["description"])

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "Polaridad",
            f"{result['polarity']:.3f}"
        )

    with m2:
        st.metric(
            "Subjetividad",
            f"{result['subjectivity']:.3f}"
        )

    with m3:
        st.metric(
            "Palabras",
            result["word_count"]
        )

    with m4:
        st.metric(
            "Caracteres",
            result["character_count"]
        )

    chart1, chart2 = st.columns(2)

    with chart1:

        st.markdown("### 🎯 Polaridad")

        st.plotly_chart(
            create_polarity_chart(result["polarity"]),
            use_container_width=True
        )

    with chart2:

        st.markdown("### 🧩 Subjetividad")

        st.plotly_chart(
            create_subjectivity_chart(
                result["subjectivity"]
            ),
            use_container_width=True
        )

    st.markdown("### 📝 Texto procesado")

    with st.container(border=True):

        st.write(
            f"**Original:** {result['original_text']}"
        )

        if result["translated"]:

            st.write(
                f"**Analizado:** {result['analyzed_text']}"
            )

        if result["translation_error"]:

            st.warning(
                "La traducción presentó un problema. "
                "Se utilizó el texto original."
            )


    with st.expander("🔬 Información técnica"):

        st.json(
            result,
            expanded=True
        )


if st.session_state.history:

    st.markdown("---")

    st.markdown("## 🕘 Historial de análisis")

    history_data = []

    for item in st.session_state.history:

        history_data.append({
            "Hora": item["timestamp"],
            "Sentimiento": (
                f"{item['emoji']} "
                f"{item['sentiment']}"
            ),
            "Polaridad": round(
                item["polarity"], 3
            ),
            "Subjetividad": round(
                item["subjectivity"], 3
            ),
            "Palabras": item["word_count"],
            "Texto": item["original_text"][:70]
        })

    df = pd.DataFrame(history_data)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


st.markdown("---")

st.caption(
    "🧠 SentimentAI • Procesamiento de Lenguaje Natural "
    "• TextBlob + Python + Streamlit"
)
