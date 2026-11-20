from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from graphrag_pipeline import cli
from graphrag_pipeline.api import create_app
from graphrag_pipeline.embeddings import HashingEmbedder
from graphrag_pipeline.pipeline import GraphIndex

from .conftest import requires_spacy

runner = CliRunner()


@requires_spacy
def test_build_then_query_in_every_mode(mini_corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "index"
    result = runner.invoke(cli.app, ["build", str(mini_corpus), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert {"graph.json", "graph.graphml", "communities.json", "manifest.json"} <= {p.name for p in out.iterdir()}
    local = runner.invoke(
        cli.app, ["query", "Where was the founder of Acme Robotics born?", "-i", str(out), "-e", "hashing", "--json"]
    )
    assert local.exit_code == 0, local.output
    payload = json.loads(local.output)
    assert payload["seeds"][0]["name"] == "Acme Robotics" and payload["hits"]
    path = runner.invoke(cli.app, ["query", "x", "-i", str(out), "-m", "path", "--from", "Rover X1",
                                   "--to", "Lindport", "--json"])
    assert path.exit_code == 0 and json.loads(path.output)["paths"]
    shown = runner.invoke(cli.app, ["query", "What is in Lindport?", "-i", str(out), "-m", "global", "-e", "hashing"])
    assert shown.exit_code == 0 and "community" in shown.output
    bad = runner.invoke(cli.app, ["query", "x", "-i", str(out), "-m", "sideways"])
    assert bad.exit_code != 0


def test_query_without_an_index_explains_how_to_build(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["query", "anything", "-i", str(tmp_path / "missing")])
    assert result.exit_code != 0 and "graphrag build" in result.output


def test_communities_report_and_png(sample_index: GraphIndex, tmp_path: Path) -> None:
    sample_index.save(tmp_path / "idx")
    table = runner.invoke(cli.app, ["communities", "-i", str(tmp_path / "idx"), "--level", "1"])
    assert table.exit_code == 0 and "L1-C0" in table.output
    report = runner.invoke(cli.app, ["communities", "-i", str(tmp_path / "idx"), "--show", "L0-C0",
                                     "--png", str(tmp_path / "graph.png")])
    assert report.exit_code == 0, report.output
    assert (tmp_path / "graph.png").stat().st_size > 10_000
    missing = runner.invoke(cli.app, ["communities", "-i", str(tmp_path / "idx"), "--show", "L9-C9"])
    assert missing.exit_code != 0


def test_demo_runs_offline_and_records_svg(
    sample_index: GraphIndex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "build_index", lambda *args, **kwargs: sample_index)
    svg = tmp_path / "demo.svg"
    result = runner.invoke(cli.app, ["demo", "--embedder", "hashing", "--svg", str(svg)])
    assert result.exit_code == 0, result.output
    assert "Reasoning paths" in result.output and "Path 1" in result.output and "community" in result.output
    assert svg.read_text(encoding="utf-8").startswith("<svg")


def test_api_endpoints(sample_index: GraphIndex) -> None:
    from fastapi.testclient import TestClient

    client = TestClient(create_app(sample_index, HashingEmbedder()))
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/stats").json()["stats"]["entities"] == sample_index.kg.graph.number_of_nodes()
    local = client.post("/query", json={"question": "Who founded the maker of the Wayfarer truck?", "k": 3})
    assert local.status_code == 200 and len(local.json()["hits"]) == 3
    path = client.post("/query", json={"question": "How is the Stackbot connected to Velltrix Systems?",
                                       "mode": "path"})
    assert path.json()["paths"]
    assert client.post("/query", json={"question": "Battery makers?", "mode": "global"}).json()["points"]
    assert client.get("/communities", params={"level": 0}).json()
    assert client.get("/entities/velltrix-systems").json()["entity"]["name"] == "Velltrix Systems"
    assert client.get("/entities/nobody").status_code == 404
    assert client.post("/query", json={"question": "hi"}).status_code == 422
    offline_vectors = TestClient(create_app(sample_index))
    assert offline_vectors.post("/query", json={"question": "Who makes Gridvault?"}).status_code == 400


def test_version_command() -> None:
    result = runner.invoke(cli.app, ["version"])
    assert result.exit_code == 0 and result.output.strip() == "0.1.0"
