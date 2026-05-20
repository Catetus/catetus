"""SplatBench v2 ingest harness.

Public entry points:

- ``bench_ingest.measure.measure_scene`` — core measurement function. Pure
  Python; shells out to ``catetus analyze`` / ``catetus optimize``.
- ``bench_ingest.cli.main`` — CLI shim (``python -m bench_ingest.cli ...``).
- ``bench_ingest.modal_app`` — Modal entrypoint (imported lazily so the local
  CLI does not require the ``modal`` package).
"""

__version__ = "0.1.0"
