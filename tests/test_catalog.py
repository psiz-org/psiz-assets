# -*- coding: utf-8 -*-
# Copyright 2026 The PsiZ Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import sys
import types
from pathlib import Path

import pytest

from psiz_assets import (
    AssetError,
    HuggingFaceAssetStore,
    ImplementationNotFoundError,
    LocalAssetStore,
    ManifestError,
    ResolvedModelResource,
    ResourceError,
    load,
)


def test_loads_canonical_bundle_and_alias():
    bundle = load("imagenet-1k-train-hsj@1.0.0")
    # alias = load("ilsvrc2012train@1.0.0")

    assert bundle.id == "imagenet-1k-train-hsj"
    # assert alias.id == bundle.id
    assert bundle.dataset().uri.startswith("hf://")
    assert bundle.dataset().variant == "8-rank-3"
    assert bundle.dataset(variant="2-rank-1").variant == "2-rank-1"
    assert bundle.benchmark().model_interface == "image_embedding"
    assert bundle.benchmark().dataset_resource.id == "imagenet-1k-train-hsj"
    assert bundle.benchmark().dataset_resource.variant == "8-rank-3"
    assert bundle.model().id == "hvi-un-d4"
    assert bundle.model("resnet50").provenance["adapter"] == "PyTorchResNetAdapter"
    assert bundle.model("clip_vitl14").provenance["framework"] == "torch"
    assert bundle.model(name="hvi-un-d4").uri.endswith("hvi-un-d4")


def test_model_resource_resolve_returns_resolved_wrapper(tmp_path):
    bundle = load("imagenet-1k-train-hsj@1.0.0")
    model_resource = bundle.model(name="hvi-un-d4")
    local_model = model_resource.with_model_path(tmp_path)

    assert isinstance(local_model, ResolvedModelResource)
    assert local_model.model_path == tmp_path
    assert local_model.id == model_resource.id
    assert local_model.version == model_resource.version
    assert local_model.uri == model_resource.uri


def test_benchmark_requirements_string_representation():
    bundle = load("imagenet-1k-train-hsj@1.0.0")
    req = bundle.benchmark().requirements()

    assert str(req) == (
        "Benchmark requirements\n"
        "───────────────────────\n"
        "\n"
        "evaluate\n"
        "  • image_root: ImageNet-1K images in ILSVRC2012 layout\n"
        "  • dataset: imagenet-1k-train-hsj@1.0.0\n"
        "\n"
        "compare_representations\n"
        "  • image_root: ImageNet-1K images in ILSVRC2012 layout\n"
        "  • reference_model: Canonical psychological embedding model of imagenet-1k-train-hsj@1.0.0"
    )


def test_resource_validation_requires_existing_image_root(tmp_path: Path):
    bundle = load("imagenet-1k-train-hsj@1.0.0")
    benchmark = bundle.benchmark()
    resolved_dataset = bundle.dataset().with_dataset_path(tmp_path)

    with pytest.raises(ResourceError):
        benchmark.with_resources(
            image_root=tmp_path / "missing",
            dataset=resolved_dataset,
        ).resources.validate()

    bound_benchmark = benchmark.with_resources(
        image_root=tmp_path,
        dataset=resolved_dataset,
    )
    resources = bound_benchmark.resources
    assert resources.dataset.id == "imagenet-1k-train-hsj"
    assert resources.values["n_select"] == 3
    resources.validate(mode="smoke")
    resources.validate(mode="complete")


def test_with_resources_accepts_resolved_model_resource(tmp_path: Path):
    bundle = load("imagenet-1k-train-hsj@1.0.0")
    benchmark = bundle.benchmark()
    resolved_dataset = bundle.dataset().with_dataset_path(tmp_path)
    resolved_model = bundle.model(name="hvi-un-d4").with_model_path(tmp_path / "model")

    resources = benchmark.with_resources(
        image_root=tmp_path,
        dataset=resolved_dataset,
        reference_model=resolved_model,
    ).resources

    assert resources.values["reference_model"] == resolved_model.model_path


def test_with_resources_rejects_unresolved_reference_model(tmp_path: Path):
    bundle = load("imagenet-1k-train-hsj@1.0.0")
    benchmark = bundle.benchmark()
    resolved_dataset = bundle.dataset().with_dataset_path(tmp_path)

    with pytest.raises(TypeError, match="ResolvedModelResource"):
        benchmark.with_resources(
            image_root=tmp_path,
            dataset=resolved_dataset,
            reference_model="not-resolved",  # type: ignore[arg-type]
        )


def test_invalid_identifier_and_version():
    with pytest.raises(ValueError):
        load("imagenet-1k-train-hsj")
    with pytest.raises(Exception):
        load("imagenet-1k-train-hsj@9.0.0")
    with pytest.raises(Exception):
        load("imagenet_train_hsj@1.0.0")


def test_builtin_manifest_uses_semver():
    bundle = load("imagenet-1k-train-hsj@1.0.0")

    assert bundle.version == "1.0.0"


def test_invalid_manifest_is_rejected(monkeypatch):
    import psiz_assets.catalog as catalog

    monkeypatch.setattr(
        catalog,
        "_manifest",
        lambda: {
            "id": "invalid",
            "version": "v1",
            "datasets": [],
            "benchmarks": [],
        },
    )

    with pytest.raises(ManifestError, match="semver"):
        load("invalid@v1")


def test_manifest_rejects_invalid_default_model_id(monkeypatch):
    import psiz_assets.catalog as catalog

    base_manifest = catalog._manifest()

    invalid_type = dict(base_manifest)
    invalid_type["default_model_id"] = ""

    monkeypatch.setattr(catalog, "_manifest", lambda: invalid_type)
    with pytest.raises(ManifestError, match="default_model_id"):
        load("imagenet-1k-train-hsj@1.0.0")


def test_manifest_rejects_unknown_default_model_id(monkeypatch):
    import psiz_assets.catalog as catalog

    base_manifest = catalog._manifest()

    unknown_default = dict(base_manifest)
    unknown_default["default_model_id"] = "missing-model"

    monkeypatch.setattr(catalog, "_manifest", lambda: unknown_default)
    with pytest.raises(ManifestError, match="default_model_id"):
        load("imagenet-1k-train-hsj@1.0.0")


def test_benchmark_evaluate_wraps_provider_result(monkeypatch, tmp_path):
    import psiz_assets.core as core

    class Provider:
        def evaluate(
            self,
            model,
            *,
            resources,
            batch_size,
            split,
            verbose,
            num_workers,
            pin_memory,
            encoder_batch_size,
        ):
            assert model == "model"
            assert resources.values["image_root"] == tmp_path
            assert batch_size == 1024
            assert split == "test"
            assert verbose == 0
            assert num_workers == 0
            assert pin_memory is True
            assert encoder_batch_size == 64
            return {
                "score": 0.75,
                "metrics": {"accuracy": 0.75, "cce": 0.4},
                "predictions": [{"prediction": 1}],
            }

    class EntryPoint:
        name = "imagenet_1k_train_hsj_v1"

        @staticmethod
        def load():
            return Provider

    monkeypatch.setattr(core, "entry_points", lambda group: [EntryPoint()])
    bundle = load("imagenet-1k-train-hsj@1.0.0")
    benchmark = bundle.benchmark().with_resources(
        image_root=tmp_path,
        dataset=bundle.dataset().with_dataset_path(tmp_path),
    )
    result = benchmark.evaluate(
        "model"
    )

    assert result.score == 0.75
    assert result.metrics["cce"] == 0.4
    assert result.summary()["benchmark"] == "default@1.0.0"


def test_benchmark_evaluate_forwards_runtime_controls(monkeypatch, tmp_path):
    import psiz_assets.core as core

    calls = {}

    class Provider:
        def evaluate(
            self,
            model,
            *,
            resources,
            batch_size,
            split="test",
            verbose,
            num_workers=0,
            pin_memory=True,
            encoder_batch_size=64,
        ):
            calls.update(batch_size=batch_size, verbose=verbose, split=split)
            return {"metrics": {"accuracy": 0.0}, "score": 0.0}

    class EntryPoint:
        name = "imagenet_1k_train_hsj_v1"

        @staticmethod
        def load():
            return Provider

    monkeypatch.setattr(core, "entry_points", lambda group: [EntryPoint()])
    bundle = load("imagenet-1k-train-hsj@1.0.0")
    benchmark = bundle.benchmark().with_resources(
        image_root=tmp_path,
        dataset=bundle.dataset().with_dataset_path(tmp_path),
    )
    benchmark.evaluate(
        "model",
        batch_size=16,
        split="validation",
        verbose=1,
    )

    assert calls == {"batch_size": 16, "verbose": 1, "split": "validation"}


def test_benchmark_compare_representations_wraps_provider_result(monkeypatch, tmp_path):
    import psiz_assets.core as core

    class Provider:
        def compare_representations(
            self,
            candidate_model,
            *,
            resources,
            reference_model,
            method,
            batch_size,
            verbose,
            num_workers,
            pin_memory,
            encoder_batch_size,
        ):
            assert candidate_model == "candidate"
            assert resources.values["image_root"] == tmp_path
            assert reference_model == tmp_path / "reference"
            assert method == "rsa"
            assert batch_size == 1024
            assert verbose == 0
            assert num_workers == 0
            assert pin_memory is True
            assert encoder_batch_size == 64
            return {
                "score": 0.2,
                "metrics": {"rsa": 0.2},
                "predictions": [{"pair": 1}],
            }

    class EntryPoint:
        name = "imagenet_1k_train_hsj_v1"

        @staticmethod
        def load():
            return Provider

    monkeypatch.setattr(core, "entry_points", lambda group: [EntryPoint()])
    bundle = load("imagenet-1k-train-hsj@1.0.0")
    resolved_reference_model = bundle.model(name="hvi-un-d4").with_model_path(
        tmp_path / "reference"
    )
    benchmark = bundle.benchmark().with_resources(
        image_root=tmp_path,
        dataset=bundle.dataset().with_dataset_path(tmp_path),
        reference_model=resolved_reference_model,
    )
    result = benchmark.compare_representations("candidate")

    assert result.score == 0.2
    assert result.metrics["rsa"] == 0.2
    assert result.summary()["benchmark"] == "default@1.0.0"


def test_benchmark_compare_representations_forwards_runtime_controls(monkeypatch, tmp_path):
    import psiz_assets.core as core

    calls = {}

    class Provider:
        def compare_representations(
            self,
            candidate_model,
            *,
            resources,
            reference_model,
            method,
            batch_size,
            verbose,
            num_workers=0,
            pin_memory=True,
            encoder_batch_size=64,
        ):
            calls.update(
                candidate_model=candidate_model,
                reference_model=reference_model,
                method=method,
                batch_size=batch_size,
                verbose=verbose,
                num_workers=num_workers,
                pin_memory=pin_memory,
                encoder_batch_size=encoder_batch_size,
            )
            return {"metrics": {"rsa": 0.0}, "score": 0.0}

    class EntryPoint:
        name = "imagenet_1k_train_hsj_v1"

        @staticmethod
        def load():
            return Provider

    monkeypatch.setattr(core, "entry_points", lambda group: [EntryPoint()])
    bundle = load("imagenet-1k-train-hsj@1.0.0")
    benchmark = bundle.benchmark().with_resources(
        image_root=tmp_path,
        dataset=bundle.dataset().with_dataset_path(tmp_path),
    )
    benchmark.compare_representations(
        "candidate",
        reference_model="reference",
        method="cka",
        batch_size=32,
        verbose=2,
        num_workers=1,
        pin_memory=False,
        encoder_batch_size=16,
    )

    assert calls == {
        "candidate_model": "candidate",
        "reference_model": "reference",
        "method": "cka",
        "batch_size": 32,
        "verbose": 2,
        "num_workers": 1,
        "pin_memory": False,
        "encoder_batch_size": 16,
    }


def test_benchmark_compare_representations_requires_provider_support(monkeypatch, tmp_path):
    import psiz_assets.core as core

    class Provider:
        def evaluate(
            self,
            model,
            *,
            resources,
            batch_size,
            split,
            verbose,
            num_workers,
            pin_memory,
            encoder_batch_size,
        ):
            return {"metrics": {"accuracy": 1.0}, "score": 1.0}

    class EntryPoint:
        name = "imagenet_1k_train_hsj_v1"

        @staticmethod
        def load():
            return Provider

    monkeypatch.setattr(core, "entry_points", lambda group: [EntryPoint()])
    bundle = load("imagenet-1k-train-hsj@1.0.0")
    resolved_reference_model = bundle.model(name="hvi-un-d4").with_model_path(
        tmp_path / "reference"
    )
    benchmark = bundle.benchmark().with_resources(
        image_root=tmp_path,
        dataset=bundle.dataset().with_dataset_path(tmp_path),
        reference_model=resolved_reference_model,
    )

    with pytest.raises(ImplementationNotFoundError, match="compare_representations"):
        benchmark.compare_representations("candidate")


def test_resolved_dataset_path_is_exposed_as_canonical_dir(tmp_path):
    bundle = load("imagenet-1k-train-hsj@1.0.0")
    benchmark = bundle.benchmark()
    resolved_dataset = bundle.dataset().with_dataset_path(tmp_path)

    resources = benchmark.with_resources(
        image_root=tmp_path,
        dataset=resolved_dataset,
    ).resources

    assert resources.values["canonical_dir"] == tmp_path
    resources.validate()


def test_local_asset_store_resolves_file_uri(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    resource = load("imagenet-1k-train-hsj@1.0.0").dataset()
    local_resource = resource.__class__(
        id=resource.id,
        version=resource.version,
        variant=resource.variant,
        uri=artifact.as_uri(),
        provenance=resource.provenance,
    )

    resolved = local_resource.resolve(LocalAssetStore())
    assert resolved.dataset_path == artifact
    assert resolved.id == local_resource.id
    assert resolved.variant == local_resource.variant


def test_hugging_face_asset_store_resolves_hf_uri(monkeypatch, tmp_path):
    calls = {}

    def snapshot_download(
        repo_id, repo_type=None, revision=None, cache_dir=None, allow_patterns=None
    ):
        calls.update(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
        )
        root = tmp_path / "hub" / "datasets--psiz--imagenet-1k-train-hsj"
        (root / "8-rank-3").mkdir(parents=True)
        return str(root)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=snapshot_download),
    )

    store = HuggingFaceAssetStore(cache_dir=tmp_path / "cache")
    resolved = store.resolve("hf://psiz/imagenet-1k-train-hsj/8-rank-3")

    assert resolved == tmp_path / "hub" / "datasets--psiz--imagenet-1k-train-hsj" / "8-rank-3"
    assert calls == {
        "repo_id": "psiz/imagenet-1k-train-hsj",
        "repo_type": "dataset",
        "revision": "main",
        "cache_dir": tmp_path / "cache",
        "allow_patterns": ["8-rank-3", "8-rank-3/**"],
    }


def test_hugging_face_asset_store_resolves_hf_repo_root_model_uri(monkeypatch, tmp_path):
    calls = {}

    def snapshot_download(**kwargs):
        calls.update(kwargs)
        root = tmp_path / "hub" / "models--psiz--hvi-un-d4"
        root.mkdir(parents=True)
        return str(root)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=snapshot_download),
    )

    store = HuggingFaceAssetStore(cache_dir=tmp_path / "cache")
    resolved = store.resolve("hf://psiz/hvi-un-d4")

    assert resolved == tmp_path / "hub" / "models--psiz--hvi-un-d4"
    assert calls == {
        "repo_id": "psiz/hvi-un-d4",
        "repo_type": "model",
        "revision": "main",
        "cache_dir": tmp_path / "cache",
    }


def test_hugging_face_asset_store_rejects_invalid_hf_uri():
    store = HuggingFaceAssetStore()
    with pytest.raises(AssetError):
        store.resolve("hf://psiz")


def test_hugging_face_asset_store_raises_when_subpath_missing(monkeypatch, tmp_path):
    def snapshot_download(
        repo_id, repo_type=None, revision=None, cache_dir=None, allow_patterns=None
    ):
        root = tmp_path / "hub" / "datasets--psiz--imagenet-1k-train-hsj"
        root.mkdir(parents=True)
        return str(root)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=snapshot_download),
    )
    store = HuggingFaceAssetStore()

    with pytest.raises(FileNotFoundError):
        store.resolve("hf://psiz/imagenet-1k-train-hsj/2-rank-1")
