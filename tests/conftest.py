"""Shared fixtures. Every test runs offline: the API key is removed and offline mode is forced."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings
from hypothesis.configuration import set_hypothesis_home_dir

from graphrag_pipeline.data import corpus_dir
from graphrag_pipeline.embeddings import HashingEmbedder
from graphrag_pipeline.pipeline import BuildConfig, GraphIndex, build_index


def spacy_model_available() -> bool:
    try:
        import spacy

        return bool(spacy.util.is_package("en_core_web_sm"))
    except ImportError:
        return False


# Property tests check invariants, not speed: no per-example deadline, and no failure when a busy machine
# makes input generation slow.
settings.register_profile("default", deadline=None, suppress_health_check=[HealthCheck.too_slow])
set_hypothesis_home_dir(Path(tempfile.gettempdir()) / "graphrag-pipeline-hypothesis")  # keep the repo clean
settings.load_profile("default")

requires_spacy = pytest.mark.skipif(not spacy_model_available(), reason="spaCy model en_core_web_sm not installed")


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("GRAPHRAG_OFFLINE", "1")


@pytest.fixture(scope="session")
def sample_index() -> GraphIndex:
    """The bundled corpus, built once per test session with the offline extractor."""
    if not spacy_model_available():
        pytest.skip("spaCy model en_core_web_sm not installed")
    return build_index(corpus_dir(), BuildConfig(extractor="spacy"))


@pytest.fixture(scope="session")
def hashing() -> HashingEmbedder:
    return HashingEmbedder()


MINI_DOCS = {
    "acme.md": "# Acme Robotics\n\nAcme Robotics is a robotics company headquartered in Lindport. "
    "Jane Doyle founded Acme Robotics in 2010.\n\nAcme Robotics makes the Rover X1, a delivery robot.\n",
    "jane-doyle.md": "# Jane Doyle\n\nJane Doyle was born in Fenwater. She studied physics at Lindport University.\n",
    "lindport.md": "# Lindport\n\nLindport is a port city in the province of Carro.\n",
}


@pytest.fixture()
def mini_corpus(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    folder.mkdir()
    for name, text in MINI_DOCS.items():
        (folder / name).write_text(text, encoding="utf-8")
    return folder
