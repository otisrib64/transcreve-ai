"""OCR por modelo de visao local via Ollama (alternativa opcional ao tesseract).

Ativado com `--ocr-engine ollama` ou `VIDEO_KB_OCR_ENGINE=ollama`. O tesseract
continua sendo o padrao; se o Ollama nao responder, o pipeline volta para ele.

Modelos de visao leem texto sobre video (GC, placar, legendas queimadas, telas
de software) muito melhor que o tesseract. Benchmark numa RTX 2080 Ti
(Ollama 0.34): em imagens limpas glm-ocr (~280 tok/s, ~4 GB de VRAM) e
deepseek-ocr:3b (~245 tok/s, ~9 GB) erram <1% dos caracteres; em frames reais
de video o deepseek-ocr inventa texto (tabelas, outro idioma) em frames de
transicao e o glm-ocr le fielmente, por isso ele e o padrao. O glm-ocr repete a
saida as vezes (ver _STOPS e clean_ocr_output).
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from .ocr import OcrResult

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OCR_MODEL = "glm-ocr"

_GENERIC_PROMPT = (
    "Transcreva todo o texto visivel nesta imagem exatamente como aparece, linha "
    "por linha, mantendo acentos, numeros e pontuacao. Responda somente com o "
    "texto. Se nao houver texto legivel, responda com uma linha vazia."
)
# Modelos especializados em OCR respondem melhor ao prompt de treino deles.
_PROMPTS = {"glm-ocr": "Text Recognition:", "deepseek-ocr": "Free OCR."}
# glm-ocr no Ollama nao registra <|user|> como fim de geracao e repete o texto
# indefinidamente (ollama/ollama#16892). A repeticao comeca sempre com uma cerca
# de codigo, entao ela serve de parada.
_STOPS = {"glm-ocr": ["<|user|>", "```"]}
# Modelos genericos (qwen3.5, gemma4...) gastam o limite pensando se deixados.
_OCR_ONLY_MODELS = set(_PROMPTS)

_GROUNDING_TAG = re.compile(r"<\|(?:ref|det|/ref|/det)\|>.*?(?=<\||$)|<\|[a-z_/]+\|>", re.S)
_FENCE = re.compile(r"^```[a-z]*\s*$", re.M)
_HTML_TAG = re.compile(r"<[^>]+>")
# Frases que o modelo escreve quando nao ha texto legivel (nao sao OCR).
_NO_TEXT = re.compile(
    r"(the image (provided )?(is|appears)|no (discernible|legible|readable) text|"
    r"i will provide a description|nao ha texto|não há texto|sem texto legivel)",
    re.I,
)


def ollama_url() -> str:
    return os.environ.get("VIDEO_KB_OLLAMA_URL", DEFAULT_OLLAMA_URL).rstrip("/")


def ollama_ocr_model() -> str:
    return os.environ.get("VIDEO_KB_OCR_MODEL", DEFAULT_OCR_MODEL)


def ollama_available(
    url: str | None = None, model: str | None = None, timeout: float = 3.0
) -> tuple[bool, str]:
    """Confere se o servidor responde e se o modelo esta baixado."""
    base = url or ollama_url()
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=timeout) as resp:
            tags = json.load(resp)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return False, f"Ollama indisponivel em {base}: {exc}"
    if model:
        names = {m.get("name") for m in tags.get("models", [])} | {
            m.get("model") for m in tags.get("models", [])
        }
        wanted = {model, f"{model}:latest"}
        if not names & wanted:
            return False, f"Modelo '{model}' nao encontrado no Ollama (rode: ollama pull {model})"
    return True, ""


def _cut_repetition(lines: list[str]) -> list[str]:
    """Corta no inicio da primeira repeticao de bloco (em qualquer ponto da saida)."""
    n = len(lines)
    for start in range(n):
        for size in range(1, (n - start) // 2 + 1):
            block = lines[start : start + size]
            nxt = lines[start + size : start + 2 * size]
            # Linha unica so conta como loop se repetir 3+ vezes seguidas.
            if size == 1 and lines[start + 1 : start + 3] != block * 2:
                continue
            if nxt == block or (
                start + 2 * size >= n - 1 and nxt[:-1] == block[: len(nxt) - 1] and len(nxt) > 1
            ):
                return lines[: start + size]
    return lines


def _table_to_text(html: str) -> str:
    rows = re.split(r"</tr>", html, flags=re.I)
    out = []
    for row in rows:
        cells = [c.strip() for c in re.split(r"</t[dh]>", row, flags=re.I)]
        cells = [_HTML_TAG.sub("", c).strip() for c in cells]
        cells = [c for c in cells if c]
        if cells:
            out.append(" | ".join(cells))
    return "\n".join(out)


def clean_ocr_output(text: str) -> str:
    """Remove cercas de codigo, tags, tabelas duplicadas, frases de 'sem texto' e repeticoes."""
    text = _GROUNDING_TAG.sub("", text)
    text = _FENCE.sub("", text)
    # glm-ocr costuma escrever o texto e depois repetir o mesmo conteudo como <table>.
    pos = text.lower().find("<table")
    if pos > 0 and text[:pos].strip():
        text = text[:pos]
    elif pos == 0:
        text = _table_to_text(text)
    text = _HTML_TAG.sub("", text)
    lines = [line.rstrip() for line in text.splitlines()]
    useful = [line for line in lines if line.strip()]
    if useful and _NO_TEXT.search(" ".join(useful[:2])):
        return ""
    return "\n".join(_cut_repetition(useful)).strip()


def ocr_image_ollama(
    image_path: Path,
    model: str | None = None,
    url: str | None = None,
    timeout: float = 120.0,
) -> OcrResult:
    """Transcreve o texto de um frame com um modelo de visao do Ollama.

    Levanta OSError/URLError se o servidor falhar; quem chama decide o fallback.
    """
    model = model or ollama_ocr_model()
    base_name = model.split(":")[0]
    options: dict = {"temperature": 0, "num_predict": 1024, "num_ctx": 8192}
    if base_name in _STOPS:
        options["stop"] = _STOPS[base_name]
    payload: dict = {
        "model": model,
        "prompt": _PROMPTS.get(base_name, _GENERIC_PROMPT),
        "images": [base64.b64encode(image_path.read_bytes()).decode()],
        "stream": False,
        "keep_alive": os.environ.get("VIDEO_KB_OLLAMA_KEEP_ALIVE", "5m"),
        "options": options,
    }
    if base_name not in _OCR_ONLY_MODELS:
        payload["think"] = False
    req = urllib.request.Request(
        f"{url or ollama_url()}/api/generate",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.load(resp)
    raw = clean_ocr_output(body.get("response", ""))
    text = "\n".join(line.strip() for line in raw.splitlines() if line.strip())
    return OcrResult(text, raw)


def unload_model(model: str | None = None, url: str | None = None) -> None:
    """Libera a VRAM do modelo de OCR (a transcricao na GPU vem logo depois)."""
    payload = {"model": model or ollama_ocr_model(), "keep_alive": 0}
    req = urllib.request.Request(
        f"{url or ollama_url()}/api/generate",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30):
            pass
    except (urllib.error.URLError, OSError):
        pass


def frame_is_blank(image_path: Path, min_stddev: float = 12.0) -> bool:
    """Frame quase uniforme (preto, fade, tela lisa): nao vale mandar para o OCR.

    Modelos de visao inventam texto nesses frames. Sem Pillow devolve False.
    """
    try:
        from PIL import Image, ImageStat
    except ImportError:
        return False
    try:
        with Image.open(image_path) as img:
            return ImageStat.Stat(img.convert("L")).stddev[0] < min_stddev
    except OSError:
        return False


def frame_signature(image_path: Path) -> int | None:
    """dHash de 256 bits do frame, para achar frames praticamente iguais.

    Requer Pillow (extra `ollama`); sem ele devolve None e nada e deduplicado.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(image_path) as img:
            small = img.convert("L").resize((17, 16), Image.BILINEAR)
            px = list(small.getdata())
    except OSError:
        return None
    bits = 0
    for row in range(16):
        for col in range(16):
            left, right = px[row * 17 + col], px[row * 17 + col + 1]
            bits = (bits << 1) | (left > right)
    return bits


def same_frame(a: int | None, b: int | None, max_distance: int = 6) -> bool:
    if a is None or b is None:
        return False
    return (a ^ b).bit_count() <= max_distance
