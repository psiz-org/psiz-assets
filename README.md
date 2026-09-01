# PsiZ Assets

`psiz-assets` provides versioned manifests and public access objects for
published PsiZ datasets, models, and benchmarks.
In plain terms, it makes it easier to find and use the canonical PsiZ assets
that belong together.

Full documentation lives in `docs/` and is Read the Docs compatible.

- Start here: `docs/getting-started.rst`
- Detailed guides: `docs/guides/`
- API reference: `docs/api/`

## Organization

Datasets and models are accessible from Hugging Face repositories.

                         psiz-assets
                             │
              ┌──────────────┼──────────────┐
              │              │              │
           datasets        models       benchmarks
              │              │              │
       versioned assets  versioned assets  manifests
              │              │              │
              ▼              ▼              ▼
         Hugging Face   Hugging Face     psiz-assets
         repositories   repositories     registry
              │              │
              └──────┬───────┘
                     │
                     ▼
               HF Collections

Catalog loading resolves manifest metadata only. Dataset and model URIs are
materialized later by storage backends.
A bundle is the release unit: one named, versioned package of compatible
dataset, model, and benchmark descriptors designed to work together.

## Quickstart

```python
import psiz_assets
from psiz_assets import HuggingFaceAssetStore

bundle = psiz_assets.load("imagenet-1k-train-hsj@1.0.0")
store = HuggingFaceAssetStore()
resolved_dataset = bundle.dataset().resolve(store)
resolved_model = bundle.model().resolve(store)

bound_benchmark = bundle.benchmark().with_resources(
    image_root="/data/ilsvrc2012",
    dataset=resolved_dataset,
    reference_mode=resolved_model,
)
result = bound_benchmark.evaluate(model)
```

## Dataset selection

Choose dataset by variant and optional semver pin:

```python
dataset = bundle.dataset()  # latest default variant
dataset = bundle.dataset(variant="8-rank-3")
dataset = bundle.dataset(variant="2-rank-1")
dataset = bundle.dataset(version="1.0.0")
```

See `docs/guides/dataset-workflow.rst` for full workflow and resolution behavior.

## Benchmark usage

```python
benchmark = bundle.benchmark(version="1.0.0")

bound_benchmark = benchmark.with_resources(
    image_root="/data/ilsvrc2012",
    dataset=resolved_dataset,
    reference_mode=resolved_model,
)

eval_result = bound_benchmark.evaluate(model, split="test")
compare_result = bound_benchmark.compare_representations(candidate_model, method="rsa")
```

The `split` argument is interpreted by the benchmark provider implementation.
Common values are `"test"`, `"validation"`, and `"train"`.

See `docs/guides/benchmark-workflow.rst` for advanced options, alternate
variants, and comparison details.

## Model descriptors

```python
model_resource = bundle.model("resnet50")
model_resource.provenance["adapter"]
```

Descriptors identify adapter/framework provenance. They do not download
checkpoints or construct models during catalog lookup.

`dataset_path` points to a local canonical artifact directory. The manifest's
dataset URI describes the published resource and does not download it during
catalog loading. `image_root` must contain the ImageNet-1K files in the
standard ILSVRC2012 layout.

Catalog loading is independent of optional ML frameworks. Benchmark providers
are discovered lazily through the `psiz.benchmarks` Python entry-point group.

## Asset storage

Resource descriptors are resolved through explicit storage backends.

```python
from psiz_assets import DatasetResource, HuggingFaceAssetStore, LocalAssetStore

bundle = psiz_assets.load("imagenet-1k-train-hsj@1.0.0")

# Resolve a local file:// URI.
local_dataset = DatasetResource(
       id="imagenet-1k-train-hsj",
       version="1.0.0",
       variant="8-rank-3",
       uri="file:///data/checkpoints/imagenet-1k-train-hsj/8-rank-3",
       provenance={"format": "parquet"},
)
resolved_local_dataset = local_dataset.resolve(LocalAssetStore())
local_path = resolved_local_dataset.dataset_path

# Resolve an hf:// URI from Hugging Face.
resolved_hf_dataset = bundle.dataset().resolve(
       HuggingFaceAssetStore()
)
hf_path = resolved_hf_dataset.dataset_path
```

The Hugging Face backend imports `huggingface_hub` only when `resolve()` is
called. Local `file://` URIs can be resolved with `LocalAssetStore`.

See `docs/guides/asset-storage.rst` for complete storage details.
