Getting Started
===============

Install
-------

.. code-block:: bash

   pip install psiz-assets

Quick Example
-------------

.. code-block:: python

   import psiz_assets
   from psiz_assets import HuggingFaceAssetStore

   bundle = psiz_assets.load("imagenet-1k-train-hsj@1.0.0")
   store = HuggingFaceAssetStore()

   resolved_dataset = bundle.dataset().resolve(store)
   bound_benchmark = bundle.benchmark().with_resources(
       image_root="/data/ilsvrc2012",
       dataset=resolved_dataset,
   )

   result = bound_benchmark.evaluate(model)

Next Steps
----------

- See :doc:`guides/dataset-workflow` for variant and version selection.
- See :doc:`guides/benchmark-workflow` for evaluate and compare workflows.
- See :doc:`guides/asset-storage` for ``file://`` and ``hf://`` resolution details.
