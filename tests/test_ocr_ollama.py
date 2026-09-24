from pathlib import Path

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


def test_same_frame_threshold(tmp_path: Path):
    from PIL import Image

    a = Image.new("L", (640, 360), 255)
    b = a.copy()
    for x in range(10):  # 10 pixels mudaram: ainda a mesma tela
        b.putpixel((x, 0), 0)
    c = a.copy()
    for x in range(100):  # 100 pixels (uma linha de texto nova): tela diferente
        c.putpixel((x, 5), 0)
    assert ocr_ollama.same_frame(a, b)
    assert not ocr_ollama.same_frame(a, c)
    assert not ocr_ollama.same_frame(None, a)


def test_same_frame_detects_new_line_of_code(tmp_path: Path):
    from PIL import Image, ImageDraw

    antes = Image.new("RGB", (1280, 720), "white")
    ImageDraw.Draw(antes).text((20, 20), ">>> any([False, True])", fill="black")
    depois = antes.copy()
    ImageDraw.Draw(depois).text((20, 40), ">>> any([0, [], ''])", fill="black")
    antes.save(tmp_path / "a.png")
    depois.save(tmp_path / "b.png")
    sa = ocr_ollama.frame_signature(tmp_path / "a.png")
    sb = ocr_ollama.frame_signature(tmp_path / "b.png")
    assert not ocr_ollama.same_frame(sa, sb)


def test_ocr_frames_rereads_static_screen_every_10_seconds(tmp_path: Path, monkeypatch):
    frames = [_png(tmp_path / f"frame_{i:04d}_{i:05d}s00.png", "a") for i in range(25)]
    calls = []
    monkeypatch.setattr(
        ocr_ollama,
        "ocr_image_ollama",
        lambda p, m=None: calls.append(p.name) or OcrResult("x", "x"),
    )
    ocr_ollama.ocr_frames(frames, "m", workers=1)
    assert len(calls) == 3  # 0 s, 10 s, 20 s


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


class _FakeStream:
    """Resposta NDJSON do /api/generate em streaming."""

    def __init__(self, chunks):
        import json

        self._lines = [json.dumps(c).encode() + b"\n" for c in chunks]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self._lines)


def _patch_urlopen(monkeypatch, chunks, captured=None):
    import json

    def fake_urlopen(req, timeout):
        if captured is not None:
            captured.update(json.loads(req.data))
        return _FakeStream(chunks)

    monkeypatch.setattr(ocr_ollama.urllib.request, "urlopen", fake_urlopen)


def test_ocr_image_ollama_sends_model_prompt_and_stop(tmp_path: Path, monkeypatch):
    img = tmp_path / "f.png"
    img.write_bytes(b"\x89PNG fake")
    captured = {}
    _patch_urlopen(
        monkeypatch,
        [{"response": "Texto\n```\n"}, {"response": "Texto\nTexto", "done": True}],
        captured,
    )
    result = ocr_ollama.ocr_image_ollama(img, model="glm-ocr", url="http://x")
    assert result == OcrResult("Texto", "Texto")
    assert captured["prompt"] == "Text Recognition:"
    assert captured["stream"] is True
    assert "```" in captured["options"]["stop"]
    assert "think" not in captured


def test_ocr_image_ollama_keeps_partial_text_when_ollama_aborts_repetition(
    tmp_path: Path, monkeypatch
):
    img = tmp_path / "f.png"
    img.write_bytes(b"x")
    _patch_urlopen(
        monkeypatch,
        [
            {"response": "SPEC.md\nCONTRACT.md\n"},
            {"response": "F F F F F F F F"},
            {"error": "prediction aborted, token repeat limit reached"},
        ],
    )
    result = ocr_ollama.ocr_image_ollama(img, model="glm-ocr", url="http://x")
    assert result.text == "SPEC.md\nCONTRACT.md\nF"


def test_ocr_image_ollama_raises_model_error_without_text(tmp_path: Path, monkeypatch):
    import pytest

    img = tmp_path / "f.png"
    img.write_bytes(b"x")
    _patch_urlopen(monkeypatch, [{"error": "model failed to load"}])
    with pytest.raises(ocr_ollama.OllamaModelError):
        ocr_ollama.ocr_image_ollama(img, model="glm-ocr", url="http://x")


def test_ocr_frames_model_errors_do_not_abandon_ollama(tmp_path: Path, monkeypatch):
    frames = [_png(tmp_path / f"{i}.png", "a" if i % 2 else "b") for i in range(8)]

    def model_error(path, model=None):
        raise ocr_ollama.OllamaModelError("prediction aborted")

    monkeypatch.setattr(ocr_ollama, "ocr_image_ollama", model_error)
    monkeypatch.setattr(ocr_ollama.time, "sleep", lambda s: None)
    batch = ocr_ollama.ocr_frames(frames, "m", workers=1)
    assert batch.error == ""
    assert len(batch.failed) == 8


def test_clean_ocr_output_collapses_repeated_token_in_line():
    assert ocr_ollama.clean_ocr_output("SPEC.md F F F F F F F") == "SPEC.md F"
    assert ocr_ollama.clean_ocr_output("1 2 3 4 5") == "1 2 3 4 5"


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
    monkeypatch.setattr(ocr_ollama.time, "sleep", lambda s: None)
    batch = ocr_ollama.ocr_frames(frames, "m", workers=1)
    assert batch.results == [None, None]
    assert batch.failed == ["1.png", "2.png"]
    assert batch.error == ""  # poucas falhas: nao desiste do Ollama


def test_ocr_frames_isolated_failure_is_retried_then_skipped(tmp_path: Path, monkeypatch):
    frames = [_png(tmp_path / "1.png", "a"), _png(tmp_path / "2.png", "b")]
    calls = {"1.png": 0}

    def flaky(path, model=None):
        if path.name == "1.png":
            calls["1.png"] += 1
            if calls["1.png"] == 1:
                raise OSError("HTTP Error 500")
        return OcrResult(f"ok {path.name}", f"ok {path.name}")

    monkeypatch.setattr(ocr_ollama, "ocr_image_ollama", flaky)
    monkeypatch.setattr(ocr_ollama.time, "sleep", lambda s: None)
    batch = ocr_ollama.ocr_frames(frames, "m", workers=1)
    assert [r.text for r in batch.results] == ["ok 1.png", "ok 2.png"]
    assert batch.failed == [] and calls["1.png"] == 2


def test_ocr_frames_gives_up_after_consecutive_failures(tmp_path: Path, monkeypatch):
    frames = [_png(tmp_path / f"{i}.png", "a" if i % 2 else "b") for i in range(8)]

    def boom(path, model=None):
        raise OSError("conexao recusada")

    monkeypatch.setattr(ocr_ollama, "ocr_image_ollama", boom)
    monkeypatch.setattr(ocr_ollama.time, "sleep", lambda s: None)
    batch = ocr_ollama.ocr_frames(frames, "m", workers=1)
    assert "seguidas" in batch.error and "conexao recusada" in batch.error
