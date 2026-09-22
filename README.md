# 🧠 SentimentAI

Aplicación web para analizar el sentimiento de textos mediante Procesamiento de Lenguaje Natural (PLN). Ofrece una interfaz interactiva creada con Streamlit, métricas visuales y un historial temporal de los análisis de la sesión.

SentimentAI puede analizar textos en inglés directamente y, de forma opcional, traducir textos en español al inglés antes del análisis. La traducción prioriza un modelo local de Argos Translate, por lo que puede funcionar sin conexión una vez instalado el modelo.

## Características

- Clasificación del sentimiento como **positivo**, **neutral** o **negativo**.
- Cálculo de polaridad en el rango de `-1` a `1`.
- Cálculo de subjetividad en el rango de `0` a `1`.
- Detección automática del idioma del texto.
- Traducción opcional de español a inglés.
- Prioridad de traducción local con Argos Translate y alternativas en línea mediante `deep-translator`.
- Respaldo léxico local para expresiones en español y para textos en inglés que TextBlob clasifique como neutrales.
- Medidores interactivos de polaridad y subjetividad.
- Conteo de palabras y caracteres.
- Historial de análisis durante la sesión actual y controles para limpiar texto e historial.

## Tecnologías

| Herramienta | Uso |
| --- | --- |
| [Python](https://www.python.org/) | Lenguaje principal |
| [Streamlit](https://streamlit.io/) | Interfaz web |
| [TextBlob](https://textblob.readthedocs.io/) | Análisis de sentimiento en inglés |
| [Argos Translate](https://www.argosopentech.com/) | Traducción local español → inglés |
| `deep-translator` | Alternativas de traducción en línea |
| `langdetect` | Detección de idioma |
| Pandas | Tabla del historial |
| Plotly | Indicadores y gráficas |

## Requisitos

- Python 3.10 o superior.
- `pip` actualizado.
- Conexión a Internet solo si se desea descargar el modelo de Argos Translate o usar los servicios de traducción alternativos.

> La aplicación se ha probado en este proyecto con Python 3.14.7.

## Instalación

1. Abre una terminal en la carpeta del proyecto.

2. Crea y activa un entorno virtual (recomendado).

   **PowerShell (Windows):**

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **CMD (Windows):**

   ```bat
   py -m venv .venv
   .venv\Scripts\activate.bat
   ```

3. Instala las dependencias:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. (Recomendado) Instala el modelo local español → inglés de Argos Translate. Así la opción de traducción puede funcionar sin conexión:

   ```powershell
   python -c "import argostranslate.package as p; p.update_package_index(); package=next(x for x in p.get_available_packages() if x.from_code == 'es' and x.to_code == 'en'); p.install_from_path(package.download()); print('Modelo español→inglés instalado')"
   ```

   Este paso descarga el modelo una única vez. La aplicación guarda sus datos de Argos en `.argos-data`, `.argos-config` y `.argos-cache` dentro del proyecto, no en el perfil global del sistema.

## Ejecutar la aplicación

Con el entorno virtual activado, ejecuta:

```powershell
streamlit run main.py
```

Streamlit mostrará una dirección local, normalmente `http://localhost:8501`. Ábrela en el navegador para usar la aplicación.

Para detener el servidor, presiona `Ctrl + C` en la terminal.

## Cómo usar SentimentAI

1. Escribe o pega el contenido en el área **“Introduce el texto que deseas analizar”**.
2. Si el texto está en español, activa **“Traducir español → inglés”** en la barra lateral para intentar una traducción previa.
3. Presiona **“Analizar sentimiento”**.
4. Revisa la clasificación, métricas, medidores y el texto procesado.
5. Consulta los análisis realizados en la tabla **Historial de análisis**. El historial solo existe mientras la sesión de Streamlit permanezca abierta.

### Interpretar los resultados

| Métrica | Interpretación |
| --- | --- |
| **Positivo** | Polaridad mayor que `0.1` |
| **Neutral** | Polaridad entre `-0.1` y `0.1` (inclusive) |
| **Negativo** | Polaridad menor que `-0.1` |
| **Polaridad** | Intensidad y dirección de la emoción: `-1` muy negativa, `0` neutral y `1` muy positiva |
| **Subjetividad** | Grado de opinión personal: `0` más objetiva y `1` más subjetiva |

## Funcionamiento interno

El flujo de cada análisis es el siguiente:

1. Se valida el texto y se detecta su idioma con `langdetect`.
2. Si se habilitó la traducción, se intenta traducir de español a inglés con Argos Translate.
3. Si Argos no está disponible, se prueban Google Translator y MyMemory Translator.
4. El texto en inglés se analiza con TextBlob.
5. Para español sin traducción —o si una traducción falla— se usa un lexicón local básico de términos positivos y negativos.
6. Si TextBlob devuelve polaridad cero en inglés, se intenta un lexicón de respaldo en inglés.
7. Se guardan el resultado y la hora en el historial de la sesión.

## Estructura del proyecto

```text
SentimentAI/
├── main.py            # Aplicación Streamlit y lógica de análisis
├── requirements.txt   # Dependencias de Python
├── README.md          # Documentación del proyecto
├── .argos-data/       # Modelos locales de Argos Translate (generado)
├── .argos-config/     # Configuración local de Argos (generado)
└── .argos-cache/      # Caché local de Argos (generado)
```

## Solución de problemas

### PowerShell no permite activar el entorno virtual

Ejecuta este comando solo para la terminal actual y vuelve a activarlo:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### La traducción falla

- Verifica que el modelo de Argos Translate esté instalado con el comando de la sección de instalación.
- Si no existe el modelo local, la app intentará servicios de traducción en línea; estos requieren Internet y pueden no estar disponibles temporalmente.
- Si todos los intentos fallan, la aplicación conserva el texto original y muestra una advertencia. Para textos en español se aplica el respaldo léxico local.

### `streamlit` no se reconoce como comando

Activa el entorno virtual o usa:

```powershell
python -m streamlit run main.py
```

### Los resultados no son los esperados

El análisis de sentimiento es una estimación automática. Frases irónicas, jerga, negaciones complejas, contexto cultural, textos muy cortos y varios idiomas mezclados pueden reducir la precisión. No lo uses como única base para decisiones importantes.

## Dependencias

Las dependencias instalables están definidas en [`requirements.txt`](requirements.txt):

```text
streamlit
textblob
deep-translator
langdetect
pandas
plotly
argostranslate
```

## Licencia

Este repositorio no incluye actualmente un archivo de licencia. Añade una licencia, por ejemplo MIT, antes de distribuir o reutilizar el proyecto públicamente.
