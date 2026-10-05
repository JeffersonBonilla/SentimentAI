import streamlit as st
import requests
from textblob import TextBlob
from deep_translator import GoogleTranslator, MyMemoryTranslator
from langdetect import detect
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import re
import unicodedata



@st.cache_data(ttl=900, show_spinner=False)
def fetch_weather(location_name):
    geocoding_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location_name, "count": 1, "language": "es", "format": "json"},
        timeout=10,
    )
    geocoding_response.raise_for_status()
    locations = geocoding_response.json().get("results", [])
    if not locations:
        raise ValueError("No se encontró esa ubicación.")

    location = locations[0]
    weather_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "precipitation,weather_code,wind_speed_10m"
            ),
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "timezone": "auto",
        },
        timeout=10,
    )
    weather_response.raise_for_status()
    return location, weather_response.json()


def describe_weather_code(code):
    descriptions = {
        0: ("☀️", "Despejado"), 1: ("🌤️", "Mayormente despejado"),
        2: ("⛅", "Parcialmente nublado"), 3: ("☁️", "Nublado"),
        45: ("🌫️", "Niebla"), 48: ("🌫️", "Niebla con escarcha"),
        51: ("🌦️", "Llovizna ligera"), 53: ("🌦️", "Llovizna moderada"),
        55: ("🌧️", "Llovizna intensa"), 61: ("🌧️", "Lluvia ligera"),
        63: ("🌧️", "Lluvia moderada"), 65: ("🌧️", "Lluvia intensa"),
        80: ("🌦️", "Chubascos ligeros"), 81: ("🌧️", "Chubascos moderados"),
        82: ("⛈️", "Chubascos intensos"), 95: ("⛈️", "Tormenta eléctrica"),
        96: ("⛈️", "Tormenta con granizo ligero"),
        99: ("⛈️", "Tormenta con granizo intenso"),
    }
    return descriptions.get(code, ("🌡️", "Condición variable"))
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
    "amo", "amamos", "bueno", "buen", "buena", "buenisimo", "excelente",
    "encanta", "encanto", "feliz", "fantastico", "genial", "gusta",
    "increible", "maravilloso", "mejor", "perfecto", "positivo",
    "recomiendo", "satisfecho", "satisfecha"
}

SPANISH_NEGATIVE_WORDS = {
    "aburrido", "decepcion", "decepcionado", "decepcionante", "dolor",
    "feo", "fea", "horrible", "mal", "malo", "mala", "molesto", "negativo", "odio",
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


def normalize_token(token):
    return "".join(
        char for char in unicodedata.normalize("NFD", token.lower())
        if unicodedata.category(char) != "Mn"
    )


def analyze_word_details(text):
    """Aplica cargas léxicas, intensificadores, negación y peso adversativo."""
    tokens = re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text, flags=re.UNICODE)
    positive = {normalize_token(word) for word in SPANISH_POSITIVE_WORDS | ENGLISH_POSITIVE_WORDS}
    negative = {normalize_token(word) for word in SPANISH_NEGATIVE_WORDS | ENGLISH_NEGATIVE_WORDS}
    base_values = {word: 0.7 for word in positive}
    base_values.update({word: -0.7 for word in negative})
    base_values.update({
        "feo": -0.6, "fea": -0.6, "malo": -0.7, "mala": -0.7, "mal": -0.7,
        "bueno": 0.7, "buen": 0.7, "buena": 0.7,
        "excelente": 0.9, "perfecto": 0.9, "perfecta": 0.9,
        "horrible": -0.9, "terrible": -0.9, "odio": -1.0, "hate": -1.0,
        "love": 1.0, "amo": 1.0,
    })

    intensifiers = {"muy": 1.5, "bastante": 1.5, "extremadamente": 2.0, "sumamente": 2.0}
    negators = {"no", "nunca", "jamas"}
    normalized = [normalize_token(token) for token in tokens]
    adversatives = []
    for index, token in enumerate(normalized):
        if token == "pero":
            adversatives.append((index, index + 1))
        elif token == "sin" and index + 1 < len(tokens) and normalized[index + 1] == "embargo":
            adversatives.append((index, index + 2))

    adversative_tokens = {
        index for start, end in adversatives for index in range(start, end)
    }
    rows = []
    total = 0.0
    has_adversative = bool(adversatives)
    for index, (token, word) in enumerate(zip(tokens, normalized)):
        base = base_values.get(word, 0.0)
        modifier_parts = []
        factor = 1.0
        if base != 0.0:
            for previous in range(max(0, index - 3), index):
                if any(start <= previous < end or previous < start < index for start, end in adversatives):
                    continue
                if normalized[previous] in negators:
                    factor *= -0.5
                    modifier_parts.append(f"Negación: {tokens[previous]} × -0.5")
                    break
            for previous in range(max(0, index - 2), index):
                if any(start <= previous < end or previous < start < index for start, end in adversatives):
                    continue
                intensity = intensifiers.get(normalized[previous])
                if intensity:
                    factor *= intensity
                    modifier_parts.append(f"Intensificador: {tokens[previous]} × {intensity:.1f}")
                    break

        adjusted = base * factor
        clause_weight = 1.0
        if has_adversative and any(index < start for start, _ in adversatives):
            clause_weight = 0.4
        contribution = adjusted * clause_weight
        total += contribution
        if base > 0:
            load_type = "Positiva"
        elif base < 0:
            load_type = "Negativa"
        elif word in intensifiers:
            load_type = "Intensificador"
        elif word in negators:
            load_type = "Negación"
        elif index in adversative_tokens:
            load_type = "Conector adversativo"
        else:
            load_type = "Neutral / sin coincidencia"
        rows.append({
            "N.º": index + 1,
            "Palabra": token,
            "Carga": load_type,
            "Valor base": round(base, 2),
            "Modificador": "; ".join(modifier_parts) or "—",
            "Valor ajustado": round(adjusted, 2),
            "Peso cláusula": clause_weight,
            "Aporte": round(contribution, 2),
        })

    if total >= 1.0:
        verdict = ("Positivo alto", "Recomendado ampliamente")
    elif total >= 0.2:
        verdict = ("Positivo moderado", "Experiencia satisfactoria")
    elif total > -0.2:
        verdict = ("Neutro / mixto", "Opinión indiferente o neutra")
    elif total > -1.0:
        verdict = ("Negativo moderado", "Atención / experiencia deficiente")
    else:
        verdict = ("Negativo alto", "Alerta / experiencia crítica")
    return rows, total, verdict


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

    st.markdown("### 🌦️ Clima")
    weather_location = st.text_input("Ubicación", "San Miguel, El Salvador")

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

st.markdown("## 🌦️ Clima actual")
try:
    location_data, weather_data = fetch_weather(weather_location)
    current = weather_data["current"]
    icon, condition = describe_weather_code(current["weather_code"])
    location_label = ", ".join(
        part for part in (location_data.get("name"), location_data.get("admin1"), location_data.get("country"))
        if part
    )
    with st.container(border=True):
        st.markdown(f"### {icon} {location_label}")
        st.caption(f"{condition} · Actualizado: {current['time'].replace('T', ' ')}")
        weather_cols = st.columns(4)
        weather_cols[0].metric("Temperatura", f"{current['temperature_2m']} °C")
        weather_cols[1].metric("Sensación", f"{current['apparent_temperature']} °C")
        weather_cols[2].metric("Humedad", f"{current['relative_humidity_2m']} %")
        weather_cols[3].metric("Viento", f"{current['wind_speed_10m']} km/h")
        st.caption("Datos de [Open-Meteo](https://open-meteo.com/)")
except (requests.RequestException, ValueError, KeyError, IndexError) as error:
    st.info(f"No se pudo cargar el clima para esa ubicación: {error}")


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

    st.markdown("### Detalle tokenizado")
    st.caption(
        "Valores base léxicos: intensificadores multiplican la carga, "
        "la negación la invierte al 50 % y la primera cláusula antes de «pero» "
        "se pondera al 40 %."
    )
    token_rows, lexical_total, lexical_verdict = analyze_word_details(result["original_text"])
    if token_rows:
        st.dataframe(pd.DataFrame(token_rows), use_container_width=True, hide_index=True)
        score_col, verdict_col = st.columns([1, 3])
        score_col.metric("Puntaje léxico total", f"{lexical_total:+.2f}")
        verdict_text, verdict_label = lexical_verdict
        if lexical_total >= 0.2:
            verdict_col.success(f"{verdict_text}: {verdict_label}")
        elif lexical_total <= -0.2:
            verdict_col.error(f"{verdict_text}: {verdict_label}")
        else:
            verdict_col.info(f"{verdict_text}: {verdict_label}")
    else:
        st.info("No se encontraron palabras para mostrar.")
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
            "Texto": item["original_text"]
        })

    df = pd.DataFrame(history_data)
    display_df = df.copy()
    display_df["Texto"] = display_df["Texto"].str.slice(0, 70)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Descargar historial CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="historial_sentimentai.csv",
        mime="text/csv",
        use_container_width=True,
    )


st.markdown("---")

st.caption(
    "🧠 SentimentAI • Procesamiento de Lenguaje Natural "
    "• TextBlob + Python + Streamlit"
)



