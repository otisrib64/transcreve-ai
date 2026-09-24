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


def test_clean_ocr_output_cuts_repetition_in_the_middle():
    raw = (
        "CUPOM\n---\nItem A 10\nItem B 20\nTOTAL 30\n---\nItem A 10\nItem B 20\nTOTAL 30\n---\nItem"
    )
    assert ocr_ollama.clean_ocr_output(raw) == "CUPOM\n---\nItem A 10\nItem B 20\nTOTAL 30"


def test_clean_ocr_output_keeps_two_equal_lines():
    assert ocr_ollama.clean_ocr_output("PGM\nPGM\nPVW") == "PGM\nPGM\nPVW"


def test_clean_ocr_output_drops_duplicated_html_table():
    raw = "Nome Qtd\nCabo 2\n<table><tr><td>Nome</td><td>Qtd</td></tr></table>"
    assert ocr_ollama.clean_ocr_output(raw) == "Nome Qtd\nCabo 2"


def test_clean_ocr_output_converts_table_only_output():
    raw = "<table><tr><td>Nome</td><td>Qtd</td></tr><tr><td>Cabo</td><td>2</td></tr></table>"
    assert ocr_ollama.clean_ocr_output(raw) == "Nome | Qtd\nCabo | 2"


def test_clean_ocr_output_discards_no_text_explanations():
    raw = "The image provided is too blurry to recognize any text.\nI will provide a description."
    assert ocr_ollama.clean_ocr_output(raw) == ""


def test_frame_is_blank(tmp_path: Path):
    from PIL import Image

    preto, texto = tmp_path / "p.png", tmp_path / "t.png"
    Image.new("RGB", (64, 64), "black").save(preto)
    img = Image.new("RGB", (64, 64), "white")
    for x in range(0, 64, 3):
        for y in range(10, 20):
            img.putpixel((x, y), (0, 0, 0))
    img.save(texto)
    assert ocr_ollama.frame_is_blank(preto)
    assert not ocr_ollama.frame_is_blank(texto)


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
            return b'{"response": "Texto\\n```\\nTexto\\nTexto"}'

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


def _png(path: Path, kind: str) -> Path:
    from PIL import Image

    img = Image.new("RGB", (64, 64), "black" if kind == "blank" else "white")
    if kind != "blank":
        step = 3 if kind == "a" else 7
        for x in range(0, 64, step):
            for y in range(64):
                img.putpixel((x, y), (0, 0, 0))
    img.save(path)
    return path


def test_ocr_frames_plans_reuse_blank_and_reads_unique(tmp_path: Path, monkeypatch):
    frames = [
        _png(tmp_path / "1.png", "a"),
        _png(tmp_path / "2.png", "a"),
        _png(tmp_path / "3.png", "blank"),
        _png(tmp_path / "4.png", "b"),
    ]
    calls = []

    def fake(path, model=None):
        calls.append(path.name)
        return OcrResult(f"txt {path.name}", f"txt {path.name}")

    monkeypatch.setattr(ocr_ollama, "ocr_image_ollama", fake)
    batch = ocr_ollama.ocr_frames(frames, "m", workers=2)
    assert sorted(calls) == ["1.png", "4.png"]
    assert [r.text for r in batch.results] == ["txt 1.png", "txt 1.png", "", "txt 4.png"]
    assert (batch.reused, batch.blank, batch.error) == (1, 1, "")


def test_ocr_frames_failure_leaves_none_for_fallback(tmp_path: Path, monkeypatch):
    frames = [_png(tmp_path / "1.png", "a"), _png(tmp_path / "2.png", "b")]

    def boom(path, model=None):
        raise OSError("conexao recusada")

    monkeypatch.setattr(ocr_ollama, "ocr_image_ollama", boom)
    batch = ocr_ollama.ocr_frames(frames, "m", workers=1)
    assert batch.results == [None, None]
    assert "conexao recusada" in batch.error
