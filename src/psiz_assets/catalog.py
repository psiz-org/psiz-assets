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

"""Catalog loading and manifest validation for built-in PsiZ assets."""

from __future__ import annotations

import json
import re
from importlib.resources import files
from typing import Any

from .core import (
    Benchmark,
    RequirementDescriptor,
    BenchmarkRequirements,
    Bundle,
    DatasetResource,
    ManifestError,
    ModelResource,
)

_SEMVER_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def _manifest() -> dict[str, Any]:
    """Load the built-in manifest shipped with this package."""
    try:
        resource = files("psiz_assets").joinpath("manifests/imagenet-1k-train-hsj.json")
        return json.loads(resource.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, ModuleNotFoundError) as error:
        raise ManifestError("unable to load the built-in imagenet-1k-train-hsj manifest") from error


def load(identifier: str) -> Bundle:
    """Load a published bundle by identifier.

    Parameters
    ----------
    identifier:
        Bundle identifier in ``name@version`` form.

    Returns
    -------
    Bundle
        Parsed bundle with dataset, model, and benchmark descriptors.
    """
    name, version = _split_identifier(identifier)
    manifest = _manifest()
    _validate_manifest(manifest)
    aliases = set(manifest.get("aliases", []))
    if name != manifest.get("id") and name not in aliases:
        raise ManifestError(f"unknown bundle: {name}")
    if version != manifest.get("version"):
        raise ManifestError(f"unknown bundle version: {name}@{version}")
    return _build_bundle(manifest)


def _split_identifier(identifier: str) -> tuple[str, str]:
    try:
        name, version = identifier.rsplit("@", 1)
    except ValueError as error:
        raise ValueError("identifier must have the form 'name@version'") from error
    if not name or not version:
        raise ValueError("identifier must have the form 'name@version'")
    return name, version


def _build_bundle(manifest: dict[str, Any]) -> Bundle:
    try:
        datasets = {
            _dataset_key(item): DatasetResource(
                id=item["id"],
                version=item["version"],
                variant=item.get("variant", "default"),
                uri=item["uri"],
                provenance=item.get("provenance", {}),
            )
            for item in manifest["datasets"]
        }
        models = {
            item["id"]: ModelResource(
                id=item["id"],
                version=item["version"],
                uri=item["uri"],
                provenance=item.get("provenance", {}),
            )
            for item in manifest.get("models", [])
        }
        benchmarks = {
            item["id"]: Benchmark(
                id=item["id"],
                version=item["version"],
                implementation=item["implementation"],
                model_interface=item["protocol"]["model_interface"],
                requirements_spec=_parse_benchmark_requirements(item["requirements"]),
                protocol=item["protocol"],
                dataset_version=item["dataset_version"],
                dataset_resource=_benchmark_dataset_resource(datasets, item, manifest),
            )
            for item in manifest["benchmarks"]
        }
        return Bundle(
            id=manifest["id"],
            version=manifest["version"],
            aliases=tuple(manifest.get("aliases", [])),
            datasets=datasets,
            models=models,
            benchmarks=benchmarks,
            provenance=manifest.get("provenance", {}),
            default_dataset_variant=manifest.get("default_dataset_variant", "default"),
            default_model_id=manifest.get("default_model_id"),
        )
    except (KeyError, TypeError) as error:
        raise ManifestError("invalid bundle manifest") from error


def _dataset_key(item: dict[str, Any]) -> str:
    return f"{item['id']}::{item.get('variant', 'default')}::{item['version']}"


def _benchmark_dataset_resource(
    datasets: dict[str, DatasetResource], benchmark: dict[str, Any], manifest: dict[str, Any]
) -> DatasetResource:
    target_id = benchmark["dataset_id"]
    target_version = benchmark["dataset_version"]
    target_variant = benchmark.get(
        "dataset_variant", manifest.get("default_dataset_variant", "default")
    )
    for dataset in datasets.values():
        if (
            dataset.id == target_id
            and dataset.version == target_version
            and dataset.variant == target_variant
        ):
            return dataset
    raise ManifestError(
        "benchmark references unknown dataset resource: "
        f"{target_id}@{target_version} variant={target_variant}"
    )


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise ManifestError("bundle manifest must be an object")
    for field in ("id", "version", "datasets", "benchmarks"):
        if field not in manifest:
            raise ManifestError(f"bundle manifest is missing: {field}")
    _validate_version(manifest["version"], "bundle")
    aliases = manifest.get("aliases", [])
    if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
        raise ManifestError("bundle aliases must be a list of strings")
    if manifest["id"] in aliases:
        raise ManifestError("bundle ID cannot also be an alias")
    _validate_resources(manifest["datasets"], "dataset")
    _validate_resources(manifest.get("models", []), "model")
    benchmark_ids = set()
    for benchmark in manifest["benchmarks"]:
        if benchmark.get("id") in benchmark_ids:
            raise ManifestError(f"duplicate benchmark ID: {benchmark.get('id')}")
        benchmark_ids.add(benchmark.get("id"))
        _validate_version(benchmark.get("version"), "benchmark")
        if benchmark.get("dataset_version") not in {
            dataset["version"] for dataset in manifest["datasets"]
        }:
            raise ManifestError(
                f"benchmark references unknown dataset version: {benchmark.get('dataset_version')}"
            )
        if not benchmark.get("implementation"):
            raise ManifestError("benchmark implementation is required")
        _validate_benchmark_requirements(benchmark.get("requirements"))
    default_variant = manifest.get("default_dataset_variant", "default")
    if not isinstance(default_variant, str) or not default_variant:
        raise ManifestError("default_dataset_variant must be a non-empty string")
    dataset_variants = {dataset.get("variant", "default") for dataset in manifest["datasets"]}
    if default_variant not in dataset_variants:
        raise ManifestError(
            f"default_dataset_variant does not match a dataset variant: {default_variant}"
        )
    default_model_id = manifest.get("default_model_id")
    if default_model_id is not None:
        if not isinstance(default_model_id, str) or not default_model_id:
            raise ManifestError("default_model_id must be a non-empty string")
        model_ids = {model["id"] for model in manifest.get("models", [])}
        if default_model_id not in model_ids:
            raise ManifestError(
                f"default_model_id does not match a model id: {default_model_id}"
            )


def _validate_resources(resources: Any, kind: str) -> None:
    if not isinstance(resources, list):
        raise ManifestError(f"{kind}s must be a list")
    identifiers = set()
    for resource in resources:
        if not isinstance(resource, dict) or not resource.get("id"):
            raise ManifestError(f"each {kind} must have an ID")
        variant = resource.get("variant", "default") if kind == "dataset" else None
        if kind == "dataset" and (not isinstance(variant, str) or not variant):
            raise ManifestError("dataset variant must be a non-empty string")
        identifier = (resource["id"], resource.get("version"), variant)
        if identifier in identifiers:
            variant_msg = f" variant={variant}" if kind == "dataset" else ""
            raise ManifestError(
                f"duplicate {kind}: {resource['id']}@{resource.get('version')}{variant_msg}"
            )
        identifiers.add(identifier)
        _validate_version(resource.get("version"), kind)


def _validate_version(version: Any, kind: str) -> None:
    if not isinstance(version, str) or _SEMVER_PATTERN.fullmatch(version) is None:
        raise ManifestError(f"{kind} version must use semver: {version!r}")


def _parse_benchmark_requirements(raw: Any) -> BenchmarkRequirements:
    _validate_benchmark_requirements(raw)
    if "evaluate" in raw or "compare_representations" in raw:
        return BenchmarkRequirements(
            evaluate=tuple(_parse_requirement_items(raw.get("evaluate", []))),
            compare_representations=tuple(
                _parse_requirement_items(raw.get("compare_representations", []))
            ),
        )

    # Backward-compatible flat mapping support.
    return BenchmarkRequirements(
        evaluate=tuple(
            RequirementDescriptor(name=name, description=description)
            for name, description in raw.items()
        )
    )


def _parse_requirement_items(items: list[Any]) -> list[RequirementDescriptor]:
    return [
        RequirementDescriptor(name=item["name"], description=item["description"])
        for item in items
    ]


def _validate_benchmark_requirements(requirements: Any) -> None:
    if not isinstance(requirements, dict):
        raise ManifestError("benchmark requirements must be an object")
    if "evaluate" in requirements or "compare_representations" in requirements:
        allowed = {"evaluate", "compare_representations"}
        invalid = set(requirements) - allowed
        if invalid:
            joined = ", ".join(sorted(invalid))
            raise ManifestError(
                f"benchmark requirements include unsupported method keys: {joined}"
            )
        for method in ("evaluate", "compare_representations"):
            if method not in requirements:
                raise ManifestError(f"benchmark requirements missing method block: {method}")
            _validate_requirement_items(requirements[method], method)
        return

    # Backward-compatible flat mapping support.
    for name, description in requirements.items():
        if not isinstance(name, str) or not name:
            raise ManifestError("benchmark requirement names must be non-empty strings")
        if not isinstance(description, str) or not description:
            raise ManifestError(
                "benchmark requirement descriptions must be non-empty strings"
            )


def _validate_requirement_items(items: Any, method: str) -> None:
    if not isinstance(items, list):
        raise ManifestError(f"benchmark requirements for {method} must be a list")
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            raise ManifestError(
                f"benchmark requirements for {method} must contain objects"
            )
        name = item.get("name")
        description = item.get("description")
        if not isinstance(name, str) or not name:
            raise ManifestError(
                f"benchmark requirement names for {method} must be non-empty strings"
            )
        if not isinstance(description, str) or not description:
            raise ManifestError(
                f"benchmark requirement descriptions for {method} must be non-empty strings"
            )
        if name in seen:
            raise ManifestError(f"duplicate benchmark requirement in {method}: {name}")
        seen.add(name)
