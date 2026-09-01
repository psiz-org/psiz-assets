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

"""Core resource, benchmark, and bundle abstractions for psiz-assets."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Protocol

if TYPE_CHECKING:
    from .storage import AssetStore


def _version_key(version: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in version.split("."))


class AssetError(Exception):
    """Base exception for asset catalog and resolution errors."""

    pass


class ManifestError(AssetError):
    """Raised when bundle manifest content is invalid or unavailable."""

    pass


class ResourceError(AssetError):
    """Raised when required benchmark resources are missing or invalid."""

    pass


class ImplementationNotFoundError(AssetError):
    """Raised when a benchmark implementation entry-point is not installed."""

    pass


class BenchmarkImplementation(Protocol):
    """Protocol implemented by benchmark providers discovered via entry-points."""

    def evaluate(
        self,
        model: Any,
        *,
        resources: "BenchmarkResources",
        batch_size: int = 1024,
        split: str = "test",
        num_workers: int = 0,
        pin_memory: bool = True,
        encoder_batch_size: int = 64,
        verbose: int = 0,
    ) -> Mapping[str, Any]:
        ...

    def compare_representations(
        self,
        candidate_model: Any,
        *,
        resources: "BenchmarkResources",
        reference_model: Any | None = None,
        method: str = "rsa",
        batch_size: int = 1024,
        num_workers: int = 0,
        pin_memory: bool = True,
        encoder_batch_size: int = 64,
        verbose: int = 0,
        **values: Any,
    ) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class DatasetResource:
    """Descriptor for a published dataset resource in a bundle manifest."""

    id: str
    version: str
    variant: str
    uri: str
    provenance: Mapping[str, Any]

    def resolve(self, store: "AssetStore") -> "ResolvedDatasetResource":
        """Resolve this resource URI through a storage backend."""
        return ResolvedDatasetResource(resource=self, dataset_path=store.resolve(self.uri))

    def with_dataset_path(self, dataset_path: str | Path) -> "ResolvedDatasetResource":
        """Bind this descriptor to an already-materialized local dataset path."""
        return ResolvedDatasetResource(resource=self, dataset_path=Path(dataset_path))


@dataclass(frozen=True)
class ResolvedDatasetResource:
    """Dataset descriptor paired with a local resolved path."""

    resource: DatasetResource
    dataset_path: Path

    @property
    def id(self) -> str:
        return self.resource.id

    @property
    def version(self) -> str:
        return self.resource.version

    @property
    def variant(self) -> str:
        return self.resource.variant

    @property
    def uri(self) -> str:
        return self.resource.uri

    @property
    def provenance(self) -> Mapping[str, Any]:
        return self.resource.provenance


@dataclass(frozen=True)
class ModelResource:
    """Descriptor for a published model resource in a bundle manifest."""

    id: str
    version: str
    uri: str
    provenance: Mapping[str, Any]

    def resolve(self, store: "AssetStore") -> "ResolvedModelResource":
        """Resolve this model URI through a storage backend."""
        return ResolvedModelResource(resource=self, model_path=store.resolve(self.uri))

    def with_model_path(self, model_path: str | Path) -> "ResolvedModelResource":
        """Bind this descriptor to an already-materialized local model path."""
        return ResolvedModelResource(resource=self, model_path=Path(model_path))


@dataclass(frozen=True)
class ResolvedModelResource:
    """Model descriptor paired with a local resolved path."""

    resource: ModelResource
    model_path: Path

    @property
    def id(self) -> str:
        return self.resource.id

    @property
    def version(self) -> str:
        return self.resource.version

    @property
    def uri(self) -> str:
        return self.resource.uri

    @property
    def provenance(self) -> Mapping[str, Any]:
        return self.resource.provenance


@dataclass(frozen=True)
class RequirementDescriptor:
    """One named resource requirement and its human-readable description."""

    name: str
    description: str


@dataclass(frozen=True)
class BenchmarkRequirements:
    """Declared resources grouped by benchmark execution method."""

    evaluate: tuple[RequirementDescriptor, ...] = ()
    compare_representations: tuple[RequirementDescriptor, ...] = ()

    _REQUIREMENT_NAME_ALIASES: ClassVar[dict[str, str]] = {
        "stimulus_images": "image_root",
        "annotation_dataset": "dataset",
    }

    def __post_init__(self) -> None:
        object.__setattr__(self, "evaluate", tuple(self.evaluate))
        object.__setattr__(
            self,
            "compare_representations",
            tuple(self.compare_representations),
        )

    def _normalized_name(self, name: str) -> str:
        return self._REQUIREMENT_NAME_ALIASES.get(name, name)

    def required_names(self, method: str) -> set[str]:
        """Return normalized required resource names for a method."""
        if method == "evaluate":
            requirements = self.evaluate
        elif method == "compare_representations":
            requirements = self.compare_representations
        else:
            raise ValueError(f"unknown benchmark method: {method}")
        return {self._normalized_name(item.name) for item in requirements}

    @property
    def dataset(self) -> str | None:
        """Backward-compatible accessor for evaluate dataset description."""
        for item in self.evaluate:
            if self._normalized_name(item.name) == "dataset":
                return item.description
        return None

    @property
    def image_root(self) -> str | None:
        """Backward-compatible accessor for evaluate image root description."""
        for item in self.evaluate:
            if self._normalized_name(item.name) == "image_root":
                return item.description
        return None

    def _format_method_block(
        self, method: str, requirements: tuple[RequirementDescriptor, ...]
    ) -> list[str]:
        lines = [method]
        if not requirements:
            lines.append("  • none")
            return lines
        for requirement in requirements:
            lines.append(
                f"  • {self._normalized_name(requirement.name)}: {requirement.description}"
            )
        return lines

    def __str__(self) -> str:
        parts = [
            "Benchmark requirements",
            "───────────────────────",
            "",
            *self._format_method_block("evaluate", self.evaluate),
            "",
            *self._format_method_block(
                "compare_representations", self.compare_representations
            ),
        ]
        return "\n".join(parts)


@dataclass(frozen=True)
class BenchmarkResources:
    """Bound runtime resources supplied to benchmark implementations."""

    requirements: BenchmarkRequirements
    values: Mapping[str, Any]
    dataset: ResolvedDatasetResource | None = None

    def validate(self, mode: str = "smoke", method: str = "evaluate") -> None:
        """Validate resource values for basic path and requirement checks."""
        if mode not in {"smoke", "complete"}:
            raise ValueError("mode must be 'smoke' or 'complete'")
        required_names = self.requirements.required_names(method)
        if "dataset" in required_names and self.dataset is None:
            raise ResourceError("dataset is required")
        image_root = self.values.get("image_root")
        if "image_root" in required_names and image_root is None:
            raise ResourceError("image_root is required")
        if image_root is not None:
            path = Path(image_root)
            if not path.is_dir():
                raise ResourceError(f"image_root is not a directory: {path}")
        if "reference_model" in required_names and self.values.get("reference_model") is None:
            raise ResourceError("reference_model is required")
        canonical_dir = self.values.get("canonical_dir")
        if canonical_dir is not None and not Path(canonical_dir).is_dir():
            raise ResourceError(f"canonical_dir is not a directory: {canonical_dir}")


@dataclass(frozen=True)
class BenchmarkResult:
    """Structured output from benchmark evaluate and comparison routines."""

    score: float | None
    metrics: Mapping[str, float]
    predictions: Any
    model: Any
    benchmark: str
    dataset_version: str
    psiz_version: str | None = None
    metadata: Mapping[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        """Return a compact dictionary summary of the benchmark result."""
        return {"score": self.score, "metrics": dict(self.metrics), "benchmark": self.benchmark}

    def to_pandas(self) -> Any:
        """Convert predictions payload to a pandas DataFrame when available."""
        try:
            import pandas as pd
        except ImportError as error:
            raise ImportError("pandas is required for BenchmarkResult.to_pandas()") from error
        if self.predictions is None:
            return pd.DataFrame()
        return pd.DataFrame(self.predictions)

    def compare(self, other: "BenchmarkResult") -> dict[str, float]:
        """Compute metric deltas against another result with shared metric keys."""
        names = set(self.metrics) & set(other.metrics)
        return {name: self.metrics[name] - other.metrics[name] for name in names}


@dataclass(frozen=True)
class Benchmark:
    """Benchmark descriptor and execution entrypoint loaded from a manifest."""

    id: str
    version: str
    implementation: str
    model_interface: str
    requirements_spec: BenchmarkRequirements
    protocol: Mapping[str, Any]
    dataset_version: str
    dataset_resource: DatasetResource | None = None
    _initial_params: Mapping[str, Any] = None

    def __post_init__(self) -> None:
        if self._initial_params is None:
            object.__setattr__(self, "_initial_params", {})

    def requirements(self) -> BenchmarkRequirements:
        """Return declared benchmark requirements."""
        return self.requirements_spec

    def with_resources(
        self,
        *,
        image_root: str | Path | None = None,
        dataset: ResolvedDatasetResource,
        reference_model: ResolvedModelResource | None = None,
        validate_resources: str = "smoke",
        **values: Any,
    ) -> "BoundBenchmark":
        """Bind resolved resources and runtime parameters to this benchmark."""
        if not isinstance(dataset, ResolvedDatasetResource):
            raise TypeError(
                "dataset must be a ResolvedDatasetResource. "
                "Call bundle.dataset(...).resolve(store) first."
            )
        if reference_model is not None and not isinstance(reference_model, ResolvedModelResource):
            raise TypeError(
                "reference_model must be a ResolvedModelResource. "
                "Call bundle.model(...).resolve(store) first."
            )
        resource_values = dict(self._initial_params)
        resource_values.update(values)
        resource_values["canonical_dir"] = dataset.dataset_path
        resource_values["dataset"] = dataset
        if image_root is not None:
            resource_values["image_root"] = image_root
        if isinstance(reference_model, ResolvedModelResource):
            resource_values["reference_model"] = reference_model.model_path
        resource_values["validate_resources"] = validate_resources
        for key in ("n_select", "n_reference", "outcome_encoding"):
            if key in self.protocol:
                resource_values.setdefault(key, self.protocol[key])
        resources = BenchmarkResources(
            self.requirements_spec,
            resource_values,
            dataset=dataset,
        )
        return BoundBenchmark(benchmark=self, resources=resources)

    def _evaluate(
        self,
        model: Any,
        *,
        resources: BenchmarkResources,
        batch_size: int = 1024,
        split: str = "test",
        num_workers: int = 0,
        pin_memory: bool = True,
        encoder_batch_size: int = 64,
        verbose: int = 0,
    ) -> BenchmarkResult:
        resources.validate(method="evaluate")
        implementation = self._load_implementation()
        provider = implementation() if isinstance(implementation, type) else implementation
        result = provider.evaluate(
            model,
            resources=resources,
            batch_size=batch_size,
            split=split,
            num_workers=num_workers,
            pin_memory=pin_memory,
            encoder_batch_size=encoder_batch_size,
            verbose=verbose,
        )
        return BenchmarkResult(
            score=result.get("score"),
            metrics=result.get("metrics", {}),
            predictions=result.get("predictions"),
            model=model,
            benchmark=f"{self.id}@{self.version}",
            dataset_version=self.dataset_version,
            psiz_version=result.get("psiz_version"),
            metadata=result.get("metadata"),
        )

    def _compare_representations(
        self,
        candidate_model: Any,
        *,
        resources: BenchmarkResources,
        reference_model: Any | None = None,
        method: str = "rsa",
        batch_size: int = 1024,
        num_workers: int = 0,
        pin_memory: bool = True,
        encoder_batch_size: int = 64,
        verbose: int = 0,
        **values: Any,
    ) -> BenchmarkResult:
        resolved_reference = (
            reference_model if reference_model is not None else resources.values.get("reference_model")
        )
        if resolved_reference is not None and resources.values.get("reference_model") is None:
            updated_values = dict(resources.values)
            updated_values["reference_model"] = resolved_reference
            resources = BenchmarkResources(
                requirements=resources.requirements,
                values=updated_values,
                dataset=resources.dataset,
            )
        resources.validate(method="compare_representations")
        implementation = self._load_implementation()
        provider = implementation() if isinstance(implementation, type) else implementation
        if not hasattr(provider, "compare_representations"):
            raise ImplementationNotFoundError(
                "benchmark implementation does not support compare_representations"
            )
        result = provider.compare_representations(
            candidate_model,
            resources=resources,
            reference_model=resolved_reference,
            method=method,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            encoder_batch_size=encoder_batch_size,
            verbose=verbose,
            **values,
        )
        return BenchmarkResult(
            score=result.get("score"),
            metrics=result.get("metrics", {}),
            predictions=result.get("predictions"),
            model=candidate_model,
            benchmark=f"{self.id}@{self.version}",
            dataset_version=self.dataset_version,
            psiz_version=result.get("psiz_version"),
            metadata=result.get("metadata"),
        )

    def _load_implementation(self) -> Any:
        candidates = entry_points(group="psiz.benchmarks")
        for candidate in candidates:
            if candidate.name == self.implementation:
                return candidate.load()
        raise ImplementationNotFoundError(
            f"benchmark implementation is not installed: {self.implementation}"
        )


@dataclass(frozen=True)
class BoundBenchmark:
    """Benchmark instance with concrete resource bindings for execution."""

    benchmark: Benchmark
    resources: BenchmarkResources

    def requirements(self) -> BenchmarkRequirements:
        """Return declared requirements for the underlying benchmark."""
        return self.benchmark.requirements()

    def evaluate(
        self,
        model: Any,
        *,
        batch_size: int = 1024,
        split: str = "test",
        num_workers: int = 0,
        pin_memory: bool = True,
        encoder_batch_size: int = 64,
        verbose: int = 0,
    ) -> BenchmarkResult:
        """Run benchmark evaluation with bound resources."""
        return self.benchmark._evaluate(
            model,
            resources=self.resources,
            batch_size=batch_size,
            split=split,
            num_workers=num_workers,
            pin_memory=pin_memory,
            encoder_batch_size=encoder_batch_size,
            verbose=verbose,
        )

    def compare_representations(
        self,
        candidate_model: Any,
        *,
        reference_model: Any | None = None,
        method: str = "rsa",
        batch_size: int = 1024,
        num_workers: int = 0,
        pin_memory: bool = True,
        encoder_batch_size: int = 64,
        verbose: int = 0,
        **values: Any,
    ) -> BenchmarkResult:
        """Run representation comparison with bound resources."""
        return self.benchmark._compare_representations(
            candidate_model,
            resources=self.resources,
            reference_model=reference_model,
            method=method,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            encoder_batch_size=encoder_batch_size,
            verbose=verbose,
            **values,
        )


@dataclass(frozen=True)
class Bundle:
    """Container of published datasets, models, and benchmarks for one release."""

    id: str
    version: str
    aliases: tuple[str, ...]
    datasets: Mapping[str, DatasetResource]
    models: Mapping[str, ModelResource]
    benchmarks: Mapping[str, Benchmark]
    provenance: Mapping[str, Any]
    default_dataset_variant: str = "default"
    default_model_id: str | None = None

    def dataset(
        self,
        *,
        variant: str | None = None,
        version: str | None = None,
    ) -> DatasetResource:
        """Return a dataset descriptor by variant and optional version."""
        matches = list(self.datasets.values())
        target_variant = self.default_dataset_variant if variant is None else variant
        matches = [item for item in matches if item.variant == target_variant]
        if version is not None:
            matches = [item for item in matches if item.version == version]
        if not matches:
            raise AssetError(
                f"dataset not found: variant={target_variant!r}, version={version!r}"
            )
        return max(matches, key=lambda item: _version_key(item.version))

    def model(self, name: str | None = None, *, version: str | None = None) -> ModelResource:
        """Return a model descriptor by name and optional version pin."""
        if name is None:
            if not self.default_model_id:
                raise AssetError("model name is required because no default model is configured")
            name = self.default_model_id
        result = self.models.get(name)
        if result is None or (version is not None and result.version != version):
            raise AssetError(f"model not found: {name}@{version or '*'}")
        return result

    def benchmark(self, name: str = "default", *, version: str | None = None, **params: Any) -> Benchmark:
        """Return a benchmark descriptor by name with optional runtime params."""
        result = self.benchmarks.get(name)
        if result is None or (version is not None and result.version != version):
            raise AssetError(f"benchmark not found: {name}@{version or '*'}")
        if params:
            object.__setattr__(result, "_initial_params", params)
        return result
