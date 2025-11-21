# 🎙️ AI Voice Cloner

Ein KI-gestütztes Voice-Cloning-Tool, das Texte mit Ollama/OpenAI generiert und diese mit geklonter Stimme ausgibt.

## ✨ Features

- 🤖 **LLM-Integration**: Unterstützt Ollama (lokal) und OpenAI
- 🎭 **Voice Cloning**: Hochwertige Stimmklonierung mit Coqui TTS XTTS v2
- 🌍 **Mehrsprachig**: Unterstützt Deutsch und weitere Sprachen
- 🖥️ **Web-Interface**: Benutzerfreundliches Gradio-Interface
- 📝 **Logging**: Umfassendes Error-Handling und Logging

## 🚀 Installation

1. **Repository klonen**
```bash
cd "KI Stimme"
```

2. **Virtuelle Umgebung erstellen**
```bash
python3 -m venv venv
source venv/bin/activate  # Auf macOS/Linux
```

3. **Abhängigkeiten installieren**
```bash
pip install -r requirements.txt
```

4. **Ollama installieren** (für lokale LLM-Nutzung)
```bash
# Auf macOS mit Homebrew
brew install ollama

# Modell herunterladen
ollama pull gpt-oss:20b
```

5. **Konfiguration** (optional)
```bash
cp .env.example .env
# .env-Datei bearbeiten, falls nötig
```

## 🎯 Verwendung

1. **Ollama starten** (falls noch nicht gestartet)
```bash
ollama serve
```

2. **Anwendung starten**
```bash
python app.py
```

3. **Im Browser öffnen**: Die Anwendung öffnet sich automatisch (normalerweise unter `http://localhost:7860`)

4. **Voice Cloning**:
   - Audio-Referenz hochladen (5-10 Sekunden klare Sprache)
   - Prompt eingeben (z.B. "Erzähle eine Geschichte über einen Roboter")
   - "Stimme generieren" klicken
   - Warten bis Text generiert und Audio erzeugt wurde

## 📁 Projektstruktur

```
KI Stimme/
├── app.py              # Hauptanwendung mit Gradio UI
├── llm_handler.py      # LLM-Integration (Ollama/OpenAI)
├── voice_cloner.py     # Voice-Cloning mit Coqui TTS
├── config.py           # Zentrale Konfiguration
├── requirements.txt    # Python-Abhängigkeiten
├── .env.example        # Beispiel-Konfiguration
├── outputs/           # Generierte Audio-Dateien (automatisch erstellt)
└── README.md          # Diese Datei
```

## ⚙️ Konfiguration

Alle Einstellungen können über Umgebungsvariablen oder die `config.py` angepasst werden:

### LLM-Einstellungen
- `LLM_PROVIDER`: "ollama" oder "openai"
- `OLLAMA_BASE_URL`: URL des Ollama-Servers (Standard: http://localhost:11434/v1)
- `OLLAMA_MODEL`: Zu verwendendes Ollama-Modell (Standard: gpt-oss:20b)
- `OPENAI_API_KEY`: OpenAI API-Schlüssel (nur bei Verwendung von OpenAI)

### TTS-Einstellungen
- `TTS_MODEL`: Coqui TTS Modell (Standard: xtts_v2)
- `TTS_DEVICE`: "cpu" oder "cuda" (auf macOS immer "cpu")
- `DEFAULT_LANGUAGE`: Standardsprache (Standard: "de")

### Logging
- `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR (Standard: INFO)

## 🐛 Fehlerbehebung

### Ollama verbindet nicht
```bash
# Überprüfen ob Ollama läuft
curl http://localhost:11434/v1/models

# Ollama neustarten
ollama serve
```

### TTS-Modell lädt nicht
- Stellen Sie sicher, dass genügend RAM verfügbar ist (mindestens 4GB)
- Bei macOS: GPU (MPS) wird automatisch deaktiviert, CPU wird verwendet

### Audio-Qualität
- Verwenden Sie hochwertige Referenz-Audio-Dateien (WAV, min. 16kHz)
- 5-10 Sekunden klare Sprache ohne Hintergrundgeräusche
- Vermeiden Sie Musik oder Echos

## 📝 Logs

Alle Logs werden in der Konsole ausgegeben. Für detailliertere Logs setzen Sie `LOG_LEVEL=DEBUG` in der `.env`-Datei.

Generierte Audio-Dateien werden im `outputs/`-Ordner mit Zeitstempel gespeichert.

## 🛠️ Technologien

- **Gradio**: Web-Interface
- **Coqui TTS**: Text-to-Speech und Voice Cloning
- **Ollama**: Lokale LLM-Ausführung
- **OpenAI API**: Cloud-basierte LLM-Option
- **PyTorch**: Deep Learning Framework

## 📄 Lizenz

Dieses Projekt verwendet:
- Coqui TTS (MPL 2.0 License)
- Andere Open-Source-Komponenten gemäß ihren jeweiligen Lizenzen

## 🤝 Beitragen

Verbesserungsvorschläge und Bug-Reports sind willkommen!

## ⚠️ Hinweise

- Die erste Generierung dauert länger, da Modelle geladen werden müssen
- Voice Cloning sollte nur mit Einwilligung der Person verwendet werden
- Achten Sie auf lokale Gesetze bezüglich KI-generierter Stimmen
