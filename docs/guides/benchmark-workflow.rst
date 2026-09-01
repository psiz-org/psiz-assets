Benchmark Workflow
==================

Retrieve Bundle and Benchmark
-----------------------------

.. code-block:: python

   import psiz_assets

   bundle = psiz_assets.load("imagenet-1k-train-hsj@1.0.0")
   benchmark = bundle.benchmark()  # latest benchmark
   benchmark = bundle.benchmark(version="1.0.0")

Inspect Requirements
--------------------

.. code-block:: python

   requirements = benchmark.requirements()
   print(requirements)

``benchmark.requirements()`` returns method-scoped requirements with a
human-readable summary:

.. code-block:: text

   Benchmark requirements
   ───────────────────────

   evaluate
          • image_root: ImageNet-1K images in ILSVRC2012 layout
          • dataset: imagenet-1k-train-hsj@1.0.0

   compare_representations
          • image_root: ImageNet-1K images in ILSVRC2012 layout
          • reference_model: Canonical psychological embedding model of imagenet-1k-train-hsj@1.0.0

Resolve Required Resources
--------------------------

.. code-block:: python

   from psiz_assets import HuggingFaceAssetStore

   store = HuggingFaceAssetStore()

   resolved_dataset = bundle.dataset().resolve(store)
   resolved_model = bundle.model().resolve(store)

Bind Resources
--------------

.. code-block:: python

   bound_benchmark = benchmark.with_resources(
       image_root="/data/ilsvrc2012",
       dataset=resolved_dataset,
       reference_model=resolved_model,
       validate_resources="smoke",
   )

Evaluate
--------

.. code-block:: python

   eval_result = bound_benchmark.evaluate(
       model,
       split="test",
       batch_size=batch_size,
       num_workers=num_workers,
       encoder_batch_size=encoder_batch_size,
       verbose=verbose,
   )

The ``split`` argument is interpreted by the benchmark implementation.
Common values are ``"test"``, ``"validation"``, and ``"train"``.

Evaluate with Alternate Variant
-------------------------------

.. code-block:: python

   resolved_dataset_2rank1 = bundle.dataset(variant="2-rank-1").resolve(store)
   bound_benchmark_2rank1 = benchmark.with_resources(
       image_root="/data/ilsvrc2012",
       dataset=resolved_dataset_2rank1,
   )
   eval_result_2rank1 = bound_benchmark_2rank1.evaluate(model)

Compare Representations
-----------------------

.. code-block:: python

   compare_result = bound_benchmark.compare_representations(
       candidate_model,
       reference_model=reference_model,
       method="rsa",
   )

Provider Discovery
------------------

Catalog loading is framework-agnostic. Benchmark providers are discovered
lazily using the ``psiz.benchmarks`` Python entry-point group.
