Asset Storage
=============

Resource descriptors are resolved through explicit storage backends.

Local Storage
-------------

.. code-block:: python

   from psiz_assets import DatasetResource, LocalAssetStore

   local_dataset = DatasetResource(
       id="imagenet-1k-train-hsj",
       version="1.0.0",
       variant="8-rank-3",
       uri="file:///data/checkpoints/imagenet-1k-train-hsj/8-rank-3",
       provenance={"format": "parquet"},
   )

   resolved_local_dataset = local_dataset.resolve(LocalAssetStore())
   local_path = resolved_local_dataset.dataset_path

Hugging Face Storage
--------------------

.. code-block:: python

   import psiz_assets
   from psiz_assets import HuggingFaceAssetStore

   bundle = psiz_assets.load("imagenet-1k-train-hsj@1.0.0")
   store = HuggingFaceAssetStore()
   resolved_hf_dataset = bundle.dataset().resolve(
       store
   )
   hf_path = resolved_hf_dataset.dataset_path

The Hugging Face backend imports ``huggingface_hub`` lazily, only when
``resolve()`` is called.

Cache Location
--------------

Use ``cache_dir`` to specify a custom cache location, such as when you want Hugging Face assets on a larger local or
scratch filesystem.

.. code-block:: python

   from psiz_assets import HuggingFaceAssetStore

   # Use the default Hugging Face cache
   store = HuggingFaceAssetStore()

   # Or specify a custom cache location
   store = HuggingFaceAssetStore(cache_dir="/data/hf-cache")
