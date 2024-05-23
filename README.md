# Dis Cyril

This is a simple Alexa like that uses the [OpenWakeWord library](https://github.com/dscripka/openWakeWord) and [PortAudio](http://www.portaudio.com/) to detect the wake word "Dis Cyril".

## Requirements

- [Python 3.10](https://www.python.org/)
- [Poetry](https://python-poetry.org/)
- [PortAudio](http://www.portaudio.com/)

## Installation

1. Clone the repository

```bash
git clone git@github.com:do4-2022/dis-cyril.git
```

2. Install the dependencies

```bash
cd dis-cyril
poetry install
```

## Usage

1. Run the detector

```bash
poetry run python src/main.py
```

2. Say the wake word "Dis Cyril"

3. You can now say your command. For now it only detects weather related commands.
