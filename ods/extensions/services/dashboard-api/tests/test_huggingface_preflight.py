"""GET /api/models/huggingface/preflight and the import endpoint's gate."""

from __future__ import annotations

import json

import pytest

import hf_gguf_header
from models import GPUInfo

REPO = "org/model-GGUF"
POLICY = {
    "backendBuilds": {"nvidia": "b11429", "cpu": "b9014", "amd": "b9014"},
    "builds": {
        "b9014": {"commit": "1" * 40, "architectures": ["llama", "qwen3"]},
        "b11429": {"commit": "2" * 40, "architectures": ["llama", "nanbeige", "qwen3"]},
    },
}
GIB = 1024 ** 3


def _details(**overrides):
    details = {
        "id": REPO,
        "sha": "c" * 40,
        "contextLength": 8192,
        "contextSource": "hub_config",
        "license": "apache-2.0",
        "url": f"https://huggingface.co/{REPO}",
        "private": False,
        "gated": False,
        "pipelineTag": "text-generation",
        "tags": ["conversational"],
        "ggufArchitecture": "llama",
        "runtimeCompatible": True,
        "runtimeReason": None,
        "artifacts": [
            {
                "id": "a" * 20,
                "label": "model-Q4_K_M.gguf",
                "quantization": "Q4_K_M",
                "sizeBytes": 4 * GIB,
                "files": [{"filename": "model-Q4_K_M.gguf", "sizeBytes": 4 * GIB, "sha256": "e" * 64}],
                "installed": False,
            },
            {
                "id": "b" * 20,
                "label": "model-Q8_0.gguf",
                "quantization": "Q8_0",
                "sizeBytes": 30 * GIB,
                "files": [{"filename": "model-Q8_0.gguf", "sizeBytes": 30 * GIB, "sha256": "f" * 64}],
                "installed": False,
            },
        ],
    }
    details.update(overrides)
    return details


def _dense_header(architecture="llama", metadata=None):
    meta = {
        "general.architecture": architecture,
        "tokenizer.chat_template": "{% if tools %}<tool_call>{% endif %}{% if enable_thinking %}{% endif %}",
    }
    meta.update(metadata or {})
    return {
        "architecture": architecture, "context_length": 131072, "block_count": 32,
        "embedding_length": 4096, "attention_head_count": 32, "attention_head_count_kv": 8,
        "attention_key_length": 128, "attention_value_length": 128,
        "full_attention_interval": None, "ssm_conv_kernel": None, "ssm_inner_size": None,
        "ssm_state_size": None, "ssm_group_count": None, "ssm_time_step_rank": None,
        "bytes_read": 2_000_000, "metadata": meta,
    }


@pytest.fixture
def preflight_env(monkeypatch, tmp_path):
    import routers.models as models_router

    policy_path = tmp_path / "llama-cpp-architectures.json"
    policy_path.write_text(json.dumps(POLICY), encoding="utf-8")
    monkeypatch.setattr(models_router, "_ARCHITECTURES_PATH", policy_path)
    monkeypatch.setattr(models_router, "GPU_BACKEND", "nvidia")
    monkeypatch.setattr(models_router, "_windows_hosted_runtime", lambda: False)
    monkeypatch.setattr(models_router, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(models_router, "_bootstrap_upgrade_download_conflict", lambda: None)
    monkeypatch.setattr(models_router, "get_gpu_info", lambda: GPUInfo(
        name="NVIDIA GeForce RTX 5070 Laptop GPU", memory_used_mb=0, memory_total_mb=8192,
        memory_percent=0.0, utilization_percent=0, temperature_c=40, gpu_backend="nvidia",
    ))
    state = {"details": _details(), "header": _dense_header(), "storage": {
        "freeBytes": 20 * GIB, "totalBytes": 100 * GIB, "marginBytes": 5 * GIB}}
    reads = []
    cached_reads = []

    async def fake_details(repo_id):
        assert repo_id == REPO
        return state["details"]

    async def fake_header(repo_id, revision, filename, *, expected_size=None, token="", client=None):
        reads.append((repo_id, revision, filename, expected_size))
        header = state.get("headers", {}).get(filename, state["header"])
        if isinstance(header, Exception):
            raise header
        return header

    async def fake_storage():
        return state["storage"]

    def fake_cached(repo_id, revision, filename):
        # What the preflight read earlier in this process (the import never reads).
        cached_reads.append((repo_id, revision, filename))
        if "cached_headers" in state:
            return state["cached_headers"].get(filename)
        if not state.get("cached", True) or isinstance(state["header"], Exception):
            return None
        return state["header"] if filename == "model-Q4_K_M.gguf" else None

    monkeypatch.setattr(models_router, "_hf_repo_details", fake_details)
    monkeypatch.setattr(hf_gguf_header, "fetch_gguf_header", fake_header)
    monkeypatch.setattr(hf_gguf_header, "cached_gguf_header", fake_cached)
    monkeypatch.setattr(models_router, "_model_storage_status", fake_storage)
    state["reads"] = reads
    state["cached_reads"] = cached_reads
    return models_router, state


def _preflight(test_client):
    return test_client.get(f"/api/models/huggingface/preflight/{REPO}", headers=test_client.auth_headers)


def test_preflight_reports_header_runtime_fit_template_and_disk(test_client, preflight_env):
    _router, state = preflight_env

    response = _preflight(test_client)

    assert response.status_code == 200
    body = response.json()
    # One network header read, scoped to its artifact rather than the repository.
    assert state["reads"] == [(REPO, "c" * 40, "model-Q4_K_M.gguf", 4 * GIB)]
    assert body["header"] == {"status": "read", "file": "model-Q4_K_M.gguf", "bytesRead": 2_000_000}
    assert body["architecture"] == "llama"
    assert body["runtime"] == {"key": "nvidia", "build": "b11429", "architectureSupported": True}
    assert body["modelKind"] == "chat"
    assert body["contextLength"] == 131072
    assert body["template"] == {"present": True, "tools": True, "thinking": "toggle"}
    assert body["refusal"] is None
    assert body["artifactId"] == "a" * 20
    small, large = body["artifacts"]["a" * 20], body["artifacts"]["b" * 20]
    assert small["fit"]["estimate"] == "architecture"
    assert small["fit"]["status"] in {"fits", "fits_short_context"}
    assert large["fit"]["status"] == "too_large"
    assert large["fit"]["estimate"] == "rough"
    assert large["architecture"] is None
    assert large["template"] == {"present": None, "tools": None, "thinking": "unknown"}
    assert large["contextLength"] is None
    assert large["contextSource"] == "unavailable"
    assert small["disk"] == "ok"
    assert large["disk"] == "insufficient"


def test_preflight_refuses_an_architecture_this_build_lacks(test_client, preflight_env, monkeypatch):
    models_router, state = preflight_env
    monkeypatch.setattr(models_router, "GPU_BACKEND", "amd")
    state["header"] = _dense_header("nanbeige")

    body = _preflight(test_client).json()

    assert body["runtime"] == {"key": "amd", "build": "b9014", "architectureSupported": False}
    assert body["refusal"]["code"] == "runtime_architecture_unsupported"
    assert body["refusal"]["overridable"] is True
    assert "b9014" in body["refusal"]["message"]


def test_preflight_names_a_reranker_from_its_header(test_client, preflight_env):
    _router, state = preflight_env
    # The live Qwen3-Reranker GGUF is tagged conversational and text-ranking
    # only on the Hub; its header settles it.
    state["details"] = _details(ggufArchitecture="qwen3")
    state["header"] = _dense_header("qwen3", {"qwen3.pooling_type": 4})

    body = _preflight(test_client).json()

    assert body["modelKind"] == "reranker"
    assert body["refusal"]["code"] == "not_a_chat_model:reranker"
    assert body["refusal"]["overridable"] is False
    assert body["runtime"]["architectureSupported"] is None


def test_preflight_without_a_header_cannot_bind_a_multi_artifact_hub_hint(test_client, preflight_env):
    _router, state = preflight_env
    state["header"] = hf_gguf_header.HeaderUnavailable("rate_limited", "Hugging Face rate limit reached")
    state["storage"] = None

    body = _preflight(test_client).json()

    assert body["header"]["status"] == "unavailable"
    assert body["header"]["reason"] == "rate_limited"
    assert body["architecture"] is None
    assert body["runtime"]["architectureSupported"] is None
    assert body["template"] == {"present": None, "tools": None, "thinking": "unknown"}
    assert body["artifacts"]["a" * 20]["fit"]["estimate"] == "rough"
    assert body["artifacts"]["a" * 20]["disk"] == "unknown"
    assert body["refusal"] is None


def test_preflight_reports_where_the_context_came_from(test_client, preflight_env):
    _router, state = preflight_env

    assert _preflight(test_client).json()["contextSource"] == "gguf_header"

    state["header"] = hf_gguf_header.HeaderUnavailable("rate_limited", "Hugging Face rate limit reached")
    body = _preflight(test_client).json()
    assert body["contextLength"] is None
    assert body["contextSource"] == "unavailable"


_GATED = hf_gguf_header.HeaderUnavailable("gated", "This repository is gated")


@pytest.mark.parametrize("token, header", [
    ("", _GATED),                     # no token: the Hub will not serve the files
    ("test-read-token", _GATED),      # a token whose account has not accepted the license
])
def test_preflight_refuses_a_gated_repository_this_host_cannot_download(
    test_client, preflight_env, monkeypatch, token, header,
):
    models_router, state = preflight_env
    monkeypatch.setattr(models_router, "_hf_token", lambda: token)
    state["details"] = _details(gated=True)
    state["header"] = header

    body = _preflight(test_client).json()

    assert body["refusal"]["code"] == "gated"
    assert body["refusal"]["overridable"] is False
    assert "accept its license" in body["refusal"]["message"]
    assert "HF_TOKEN in Settings" in body["refusal"]["message"]
    assert all(row["refusal"]["code"] == "gated" for row in body["artifacts"].values())


def test_preflight_allows_a_gated_repository_the_token_can_read(test_client, preflight_env, monkeypatch):
    models_router, state = preflight_env
    monkeypatch.setattr(models_router, "_hf_token", lambda: "test-read-token")
    state["details"] = _details(gated=True)

    body = _preflight(test_client).json()

    assert body["header"]["status"] == "read"
    assert body["refusal"] is None


def test_imatrix_calibration_files_are_not_offered_as_models():
    import routers.models as models_router

    # bartowski publishes the quantization's importance matrix next to the weights.
    assert models_router._hf_supported_gguf_filename("Nanbeige_Nanbeige4.2-3B-imatrix.gguf") is False
    assert models_router._hf_supported_gguf_filename("imatrix_unsloth.gguf") is False
    assert models_router._hf_supported_gguf_filename("Nanbeige_Nanbeige4.2-3B-Q4_K_M.gguf") is True


def _import(test_client, **extra):
    return test_client.post(
        "/api/models/huggingface/import",
        headers=test_client.auth_headers,
        json={"repoId": REPO, "artifactId": "a" * 20, **extra},
    )


def test_import_refuses_an_unsupported_architecture_before_any_write(test_client, preflight_env, monkeypatch):
    models_router, state = preflight_env
    monkeypatch.setattr(models_router, "GPU_BACKEND", "amd")
    state["header"] = _dense_header("nanbeige")
    monkeypatch.setattr(models_router, "_call_agent_model", lambda *_a, **_k: pytest.fail("not dispatched"))

    response = _import(test_client)

    assert response.status_code == 422
    assert response.headers["X-ODS-Import-Started"] == "false"
    assert response.json()["detail"]["code"] == "runtime_architecture_unsupported"
    assert response.json()["detail"]["overridable"] is True
    assert not (models_router.Path(models_router.DATA_DIR) / "model-imports.json").exists()


def test_import_anyway_records_the_acknowledged_runtime(test_client, preflight_env, monkeypatch):
    models_router, state = preflight_env
    monkeypatch.setattr(models_router, "GPU_BACKEND", "amd")
    state["header"] = _dense_header("nanbeige")
    dispatched = []

    def dispatch(_path, payload):
        dispatched.append(payload)
        return {"status": "started"}

    monkeypatch.setattr(models_router, "_call_agent_model", dispatch)

    response = _import(test_client, allowUnsupportedRuntime=True)

    assert response.status_code == 200
    assert len(dispatched) == 1
    registry = json.loads((models_router.Path(models_router.DATA_DIR) / "model-imports.json").read_text(encoding="utf-8"))
    override = registry["models"][0]["runtime_override"]
    assert override["code"] == "runtime_architecture_unsupported"
    assert override["architecture"] == "nanbeige"
    assert override["build"] == "b9014"


def test_import_anyway_cannot_import_a_reranker_as_a_chat_model(test_client, preflight_env, monkeypatch):
    models_router, state = preflight_env
    state["header"] = _dense_header("qwen3", {"qwen3.pooling_type": 4})
    monkeypatch.setattr(models_router, "_call_agent_model", lambda *_a, **_k: pytest.fail("not dispatched"))

    response = _import(test_client, allowUnsupportedRuntime=True)

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "not_a_chat_model:reranker"


def test_import_record_carries_the_header_layout_and_context(test_client, preflight_env, monkeypatch):
    models_router, _state = preflight_env
    monkeypatch.setattr(models_router, "_call_agent_model", lambda path, payload: {"status": "started"})

    response = _import(test_client)

    assert response.status_code == 200
    registry = json.loads((models_router.Path(models_router.DATA_DIR) / "model-imports.json").read_text(encoding="utf-8"))
    record = registry["models"][0]
    assert record["architecture"] == "llama"
    assert record["block_count"] == 32
    assert record["attention_head_count_kv"] == 8
    assert record["recurrent_state_bytes"] == 0
    assert record["max_context_length"] == 131072
    assert record["context_source"] == "gguf_header"
    assert record["context_length"] == 32768
    assert record["template_signals"] == {"present": True, "tools": True, "thinking": "toggle"}
    assert "runtime_override" not in record


def test_search_rows_flag_text_ranking_repositories():
    import routers.models as models_router

    item = models_router._hf_search_item({
        "id": "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF",
        "pipeline_tag": "text-ranking",
        "tags": ["gguf", "text-ranking", "conversational"],
        "siblings": [{"rfilename": "qwen3-reranker-0.6b-q8_0.gguf"}],
    })

    assert item["runtimeCompatible"] is False


def test_import_never_waits_on_a_hub_header_read(test_client, preflight_env, monkeypatch):
    models_router, state = preflight_env
    monkeypatch.setattr(models_router, "_call_agent_model", lambda path, payload: {"status": "started"})

    response = _import(test_client)

    assert response.status_code == 200
    assert state["reads"] == []


def test_import_accepts_the_checked_repository_revision(test_client, preflight_env, monkeypatch):
    router, state = preflight_env
    monkeypatch.setattr(router, "_call_agent_model", lambda *_a, **_k: {"status": "started"})

    response = _import(test_client, revision="c" * 40)

    assert response.status_code == 200
    assert state["reads"] == []
    record = json.loads((router.Path(router.DATA_DIR) / "model-imports.json").read_text())["models"][0]
    assert record["source_revision"] == "c" * 40


def test_import_rejects_changed_revision_before_cache_registration_or_dispatch(test_client, preflight_env, monkeypatch):
    router, state = preflight_env
    monkeypatch.setattr(router, "_call_agent_model", lambda *_a, **_k: pytest.fail("not dispatched"))

    response = _import(test_client, revision="d" * 40, allowUnsupportedRuntime=True)

    assert response.status_code == 409
    assert response.headers["X-ODS-Import-Started"] == "false"
    assert "revision changed" in response.json()["detail"]
    assert state["reads"] == state["cached_reads"] == []
    assert not (router.Path(router.DATA_DIR) / "model-imports.json").exists()


def test_import_without_a_cached_header_gates_on_the_hub_summary(test_client, preflight_env, monkeypatch):
    models_router, state = preflight_env
    monkeypatch.setattr(models_router, "GPU_BACKEND", "amd")
    state["cached"] = False
    state["details"] = _details(ggufArchitecture="nanbeige")
    state["details"]["artifacts"] = state["details"]["artifacts"][:1]
    monkeypatch.setattr(models_router, "_call_agent_model", lambda *_a, **_k: pytest.fail("not dispatched"))

    response = _import(test_client)

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "runtime_architecture_unsupported"
    assert state["reads"] == []


def test_selected_artifact_preflight_reads_only_its_own_header(test_client, preflight_env):
    _router, state = preflight_env
    other = _dense_header("qwen3", {"tokenizer.chat_template": "plain prompt"})
    other.update(context_length=4096, block_count=24, attention_head_count_kv=4)
    state["headers"] = {"model-Q8_0.gguf": other}

    response = test_client.get(
        f"/api/models/huggingface/preflight/{REPO}?artifactId={'b' * 20}",
        headers=test_client.auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert state["reads"] == [(REPO, "c" * 40, "model-Q8_0.gguf", 30 * GIB)]
    assert body["artifactId"] == "b" * 20
    assert body["architecture"] == body["artifacts"]["b" * 20]["architecture"] == "qwen3"
    assert body["contextLength"] == 4096
    assert body["template"]["tools"] is False
    assert body["artifacts"]["a" * 20]["architecture"] == "llama"


def test_unknown_artifact_preflight_does_not_read_a_different_file(test_client, preflight_env):
    _router, state = preflight_env
    response = test_client.get(
        f"/api/models/huggingface/preflight/{REPO}?artifactId={'f' * 20}",
        headers=test_client.auth_headers,
    )
    assert response.status_code == 409
    assert state["reads"] == []


def test_selected_split_artifact_reads_its_first_part_only(test_client, preflight_env):
    _router, state = preflight_env
    artifact = state["details"]["artifacts"][1]
    artifact["files"] = [
        {"filename": f"other/model-{part:05d}-of-00002.gguf", "sizeBytes": 15 * GIB, "sha256": "f" * 64}
        for part in (1, 2)
    ]

    response = test_client.get(
        f"/api/models/huggingface/preflight/{REPO}?artifactId={'b' * 20}",
        headers=test_client.auth_headers,
    )

    assert response.status_code == 200
    assert state["reads"] == [(REPO, "c" * 40, "other/model-00001-of-00002.gguf", 15 * GIB)]
    assert response.json()["artifactId"] == "b" * 20


def test_mixed_repository_import_uses_selected_cached_header(test_client, preflight_env, monkeypatch):
    router, state = preflight_env
    other = _dense_header("qwen3", {"tokenizer.chat_template": "plain prompt"})
    other.update(context_length=4096, block_count=24, attention_head_count_kv=4)
    state["cached_headers"] = {"model-Q4_K_M.gguf": _dense_header(), "model-Q8_0.gguf": other}
    monkeypatch.setattr(router, "_call_agent_model", lambda *_a, **_k: {"status": "started"})

    response = _import(test_client, artifactId="b" * 20)

    assert response.status_code == 200
    assert state["reads"] == []
    assert state["cached_reads"] == [(REPO, "c" * 40, "model-Q8_0.gguf")]
    record = json.loads((router.Path(router.DATA_DIR) / "model-imports.json").read_text())["models"][0]
    assert record["architecture"] == "qwen3"
    assert record["block_count"] == 24
    assert record["attention_head_count_kv"] == 4
    assert record["context_length"] == record["max_context_length"] == 4096
    assert record["template_signals"]["tools"] is False


def test_unread_selected_artifact_does_not_inherit_a_cached_reranker(test_client, preflight_env, monkeypatch):
    router, state = preflight_env
    state["header"] = _dense_header("qwen3", {"qwen3.pooling_type": 4})
    monkeypatch.setattr(router, "_call_agent_model", lambda *_a, **_k: {"status": "started"})

    body = _preflight(test_client).json()
    assert body["artifacts"]["a" * 20]["refusal"]["code"] == "not_a_chat_model:reranker"
    assert body["artifacts"]["b" * 20]["refusal"] is None
    state["reads"].clear()
    response = _import(test_client, artifactId="b" * 20)

    assert response.status_code == 200
    assert state["reads"] == []
    record = json.loads((router.Path(router.DATA_DIR) / "model-imports.json").read_text())["models"][0]
    assert record["context_length"] == 8192
    assert record.get("max_context_length") is None
    assert record.get("context_source") == "unavailable"
    for key in ("architecture", "block_count", "recurrent_state_bytes", "template_signals"):
        assert key not in record


@pytest.mark.parametrize("field,value", [("repoId", "other/repo"), ("revision", "d" * 40), ("artifactId", "b" * 20), ("file", "other.gguf")])
def test_import_record_ignores_header_bound_to_another_artifact(preflight_env, field, value):
    import asyncio

    router, state = preflight_env
    artifact = state["details"]["artifacts"][0]
    gate = asyncio.run(router._hf_preflight_gate(state["details"], artifact, read_header=False))
    gate["artifact"][field] = value

    record = router._hf_import_record(state["details"], artifact, gate=gate)

    assert "architecture" not in record
    assert "template_signals" not in record
    assert "block_count" not in record
    assert record["context_length"] == 8192


def test_storage_check_allows_for_a_windows_managed_model_store(monkeypatch):
    import asyncio

    import routers.models as models_router

    seen = []

    def fake_agent(method, path, *, timeout):
        seen.append((method, path, timeout))
        return {"freeBytes": 9, "totalBytes": 10, "marginBytes": 1, "extra": "dropped"}

    monkeypatch.setattr(models_router, "request_agent_json", fake_agent)

    status = asyncio.run(models_router._model_storage_status())

    assert status == {"freeBytes": 9, "totalBytes": 10, "marginBytes": 1}
    # Strix Halo's WSL agent needs ~5 s to resolve its Windows model store;
    # the old 5 s budget dropped the disk check there on every preflight.
    assert seen == [("GET", "/v1/model/storage", models_router._MODEL_STORAGE_TIMEOUT_SECONDS)]
    assert models_router._MODEL_STORAGE_TIMEOUT_SECONDS >= 15


def _busy(operation):
    from fastapi import HTTPException

    return HTTPException(status_code=409, detail={
        "error": f"Cannot start model download while {operation} is in progress",
        "code": "model_lifecycle_busy", "activeOperation": operation, "activeTarget": "x.gguf"})


def test_import_waits_out_a_short_lifecycle_hold(test_client, preflight_env, monkeypatch):
    models_router, _state = preflight_env
    calls = []

    def agent(path, payload):
        calls.append(path)
        if len(calls) < 3:
            # The integrity check a restarted agent runs, then a Pixel re-proof.
            raise _busy("artifact_verification" if len(calls) == 1 else "pixel_access_mode")
        return {"status": "started"}

    monkeypatch.setattr(models_router, "_call_agent_model", agent)
    monkeypatch.setattr(models_router.time, "sleep", lambda _seconds: None)

    response = _import(test_client)

    assert response.status_code == 200
    assert calls == ["/v1/model/download"] * 3


def test_a_busy_host_refuses_the_import_in_words_without_starting_it(test_client, preflight_env, monkeypatch):
    models_router, _state = preflight_env
    calls = []

    def agent(path, payload):
        calls.append(path)
        raise _busy("model_activation")

    monkeypatch.setattr(models_router, "_call_agent_model", agent)

    response = _import(test_client)

    assert response.status_code == 409
    assert response.headers["X-ODS-Import-Started"] == "false"
    assert response.json()["detail"]["message"] == (
        "ODS is switching models right now, so this download cannot start yet. Try again in a minute.")
    assert len(calls) == 1  # a model switch is not a short hold


def test_a_short_hold_that_outlasts_the_grace_still_ends_in_words(test_client, preflight_env, monkeypatch):
    models_router, _state = preflight_env
    clock = [1000.0]

    def agent(path, payload):
        clock[0] += 20.0
        raise _busy("artifact_verification")

    monkeypatch.setattr(models_router, "_call_agent_model", agent)
    monkeypatch.setattr(models_router.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(models_router.time, "sleep", lambda _seconds: None)

    response = _import(test_client)

    assert response.status_code == 409
    assert "checking a downloaded model file" in response.json()["detail"]["message"]
