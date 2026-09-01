Organization
============

Datasets and models are accessible from Hugging Face repositories.

::

                           psiz-assets
                               |
                +--------------+--------------+
                |              |              |
             datasets        models       benchmarks
                |              |              |
         versioned assets  versioned assets  manifests
                |              |              |
                v              v              v
           Hugging Face   Hugging Face     psiz-assets
           repositories   repositories     registry
                |              |
                +------+-------+
                       |
                       v
                 HF Collections

Catalog loading resolves manifest metadata only. Dataset and model URIs are
resolved later by storage backends.

Terminology
-----------

- ``variant`` identifies a dataset flavor (for example, ``8-rank-3`` vs ``2-rank-1``).
- ``version`` identifies a semver-pinned publication.
