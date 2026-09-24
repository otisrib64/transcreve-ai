from pathlib import Path
from unittest import mock

from video_kb import ocr_ollama
from video_kb.ocr import OcrResult
from video_kb.pipeline import PipelineOptions


def test_clean_ocr_output_removes_fences_and_repetition():
    raw = "ON AIR\nEstudio 3\n```markdown\n\nON AIR\nEstudio 3\n```\n```\n"
    assert ocr_ollama.clean_ocr_output(raw) == "ON AIR\nEstudio 3"


def test_clean_ocr_output_trims_truncated_repeat():
    raw = "linha 1\nlinha 2\nlinha 1\nlinha 2\nlin"
    assert ocr_ollama.clean_ocr_output(raw) == "linha 1\nlinha 2"


def test_clean_ocr_output_keeps_normal_text_and_indent():
    raw = "def f():\n    return 1\n"
    assert ocr_ollama.clean_ocr_output(raw) == "def f():\n    return 1"


def test_same_frame_threshold():
    assert ocr_ollama.same_frame(0b1011, 0b1010)
    assert not ocr_ollama.same_frame(0, (1 << 20) - 1)
    assert not ocr_ollama.same_frame(None, 1)


def test_frame_signature_detects_identical_images(tmp_path: Path):
    from PIL import Image

    a, b, c = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"
    Image.new("RGB", (64, 64), "white").save(a)
    Image.new("RGB", (64, 64), "white").save(b)
    img = Image.new("RGB", (64, 64), "white")
    for x in range(0, 64, 4):
        for y in range(64):
            img.putpixel((x, y), (0, 0, 0))
    img.save(c)
    sa, sb, sc = (ocr_ollama.frame_signature(p) for p in (a, b, c))
    assert ocr_ollama.same_frame(sa, sb)
    assert not ocr_ollama.same_frame(sa, sc)


def test_ollama_available_reports_unreachable_server():
    ok, reason = ocr_ollama.ollama_available(url="http://127.0.0.1:9", timeout=0.5)
    assert not ok and "indisponivel" in reason


def test_pipeline_options_default_engine_is_tesseract(monkeypatch):
    monkeypatch.delenv("VIDEO_KB_OCR_ENGINE", raising=False)
    assert PipelineOptions(out_dir=Path("x")).ocr_engine == "tesseract"
    monkeypatch.setenv("VIDEO_KB_OCR_ENGINE", "ollama")
    assert PipelineOptions(out_dir=Path("x")).ocr_engine == "ollama"


def test_ocr_image_ollama_sends_model_prompt_and_stop(tmp_path: Path):
    img = tmp_path / "f.png"
    img.write_bytes(b"\x89PNG fake")
    captured = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"response": "Texto\\n```\\nTexto"}'

    def fake_urlopen(req, timeout):
        import json

        captured.update(json.loads(req.data))
        return Resp()

    with mock.patch.object(ocr_ollama.urllib.request, "urlopen", fake_urlopen):
        result = ocr_ollama.ocr_image_ollama(img, model="glm-ocr", url="http://x")
    assert result == OcrResult("Texto", "Texto")
    assert captured["prompt"] == "Text Recognition:"
    assert "```" in captured["options"]["stop"]
    assert "think" not in captured


def test_cli_passes_ocr_flags_only_when_given():
    import argparse

    from video_kb.cli import _ocr_option_kwargs

    assert _ocr_option_kwargs(argparse.Namespace(ocr_engine=None, ocr_model="")) == {}
    assert _ocr_option_kwargs(argparse.Namespace(ocr_engine="ollama", ocr_model="m")) == {
        "ocr_engine": "ollama",
        "ocr_model": "m",
    }
