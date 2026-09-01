Dataset Workflow
================

Load a Bundle
-------------

.. code-block:: python

   import psiz_assets

   bundle = psiz_assets.load("imagenet-1k-train-hsj@1.0.0")

Select Dataset Variant and Version
----------------------------------

.. code-block:: python

   dataset = bundle.dataset()  # latest default variant
   dataset = bundle.dataset(variant="8-rank-3")
   dataset = bundle.dataset(variant="2-rank-1")
   dataset = bundle.dataset(version="1.0.0")

Resolve and Consume
-------------------

.. code-block:: python

   from psiz_assets import HuggingFaceAssetStore

   store = HuggingFaceAssetStore()
   resolved_dataset = bundle.dataset().resolve(store)

   local_dataset_path = resolved_dataset.dataset_path

The resource URI is not downloaded during ``load()``. Downloading and local
path materialization happen only during ``resolve(store)``.
