"""Smoke tests for the worker's preset-dispatch table.

These run pure-Python and never touch Modal. The goal is to lock the
contract between :func:`enqueue` and the preset-specific Modal apps
(``splatforge-codec-gs-mixed`` and ``splatforge-fcgs``):

  1. Unrecognized presets fall through to ``run_optimize.spawn`` (the
     existing free-tier path that shells out to the splatforge CLI).
  2. Recognized presets with a configured URL get forwarded verbatim
     (job_id, preset, blob_url, filename, callback_url) and the remote
     ack is returned to the API caller.
  3. Recognized presets with no configured URL return a clear error
     synchronously, so the API marks the job ``Error`` instead of
     waiting on a callback that will never come.

Run locally with::

    cd apps/worker && python -m pytest test_worker_dispatch.py -v

The Modal SDK is imported by worker.py at module load time, so the
tests stub it out via :mod:`sys.modules` BEFORE importing worker. That
keeps the smoke layer dependency-free — no Modal account or local
``modal`` package required to run CI.
"""
from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
from unittest import mock


def _install_modal_stub() -> None:
    """Inject a minimal `modal` shim into sys.modules so `import modal`
    inside worker.py succeeds without the real SDK installed.

    The shim implements only the decorators / classes worker.py touches
    at module-import time. Real behavior is exercised in production
    only — this stub deliberately returns dummies for every call so the
    test imports are total no-ops.
    """
    if "modal" in sys.modules and getattr(sys.modules["modal"], "__splatforge_stub__", False):
        return

    modal = types.ModuleType("modal")
    modal.__splatforge_stub__ = True  # type: ignore[attr-defined]

    class _Image:
        @staticmethod
        def debian_slim(**_kw):
            return _Image()

        def apt_install(self, *_a, **_kw):
            return self

        def run_commands(self, *_a, **_kw):
            return self

        def pip_install(self, *_a, **_kw):
            return self

    class _Volume:
        @staticmethod
        def from_name(_name, create_if_missing=False):
            return _Volume()

        def commit(self):
            pass

    class _Secret:
        @staticmethod
        def from_name(_name, required_keys=None):
            return _Secret()

    class _App:
        def __init__(self, name, image=None):
            self.name = name
            self.image = image

        def function(self, *_a, **_kw):
            def _decorator(fn):
                # Attach a fake `.spawn` so call sites in worker.py
                # remain reachable in tests that exercise the
                # fallback path.
                fn.spawn = mock.MagicMock(return_value=None)  # type: ignore[attr-defined]
                return fn

            return _decorator

    def _fastapi_endpoint(*_a, **_kw):
        def _decorator(fn):
            return fn

        return _decorator

    modal.Image = _Image  # type: ignore[attr-defined]
    modal.Volume = _Volume  # type: ignore[attr-defined]
    def _asgi_app(*_a, **_kw):
        def _decorator(fn):
            return fn

        return _decorator

    modal.Secret = _Secret  # type: ignore[attr-defined]
    modal.App = _App  # type: ignore[attr-defined]
    modal.fastapi_endpoint = _fastapi_endpoint  # type: ignore[attr-defined]
    modal.asgi_app = _asgi_app  # type: ignore[attr-defined]
    sys.modules["modal"] = modal


class WorkerDispatchTests(unittest.TestCase):
    """Lock the preset → endpoint routing the API depends on."""

    def setUp(self) -> None:
        _install_modal_stub()
        # Each test starts from a clean env so the dispatch table is
        # rebuilt deterministically when we re-import worker.
        for key in (
            "SPLATFORGE_CODEC_GS_MIXED_URL",
            "SPLATFORGE_FCGS_URL",
            "SPLATFORGE_CAPTURE_URL",
            "SPLATFORGE_HACPP_LZMA_URL",
            "SPLATFORGE_HOSTED_NEURAL_URL",
            "SPLATFORGE_QAT_SCAFFOLD_URL",
            "SPLATFORGE_QAT_BUNDLE_URL",
        ):
            os.environ.pop(key, None)
        # Drop cached module so module-level `PRESET_DISPATCH_URLS`
        # picks up the current env.
        sys.modules.pop("worker", None)

    def _import_worker(self):
        # apps/worker is on the path implicitly when running pytest from
        # the directory; add it explicitly so this file also works under
        # `python -m unittest` from the repo root.
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        return importlib.import_module("worker")

    def test_local_preset_uses_run_optimize_spawn(self) -> None:
        """`web-mobile` (and any non-dispatch preset) must fall through
        to the CLI path. We verify by asserting `run_optimize.spawn`
        was called and no HTTP forward happened."""
        worker = self._import_worker()
        with mock.patch.object(worker, "_forward_to_preset_app") as forward:
            ack = worker.enqueue(
                {
                    "job_id": "00000000-0000-0000-0000-000000000001",
                    "preset": "web-mobile",
                    "blob_url": "https://blob.example/web-mobile.ply",
                    "callback_url": "https://api.example/v1/jobs/x/result",
                    "filename": "scene.ply",
                }
            )
        self.assertTrue(ack["queued"])
        self.assertIsNone(ack["error"])
        forward.assert_not_called()
        worker.run_optimize.spawn.assert_called_once()

    def test_codec_gs_mixed_without_url_returns_clear_error(self) -> None:
        """The codec-gs-mixed dispatch MUST surface a synchronous error
        when its dedicated Modal endpoint isn't configured — that's the
        only signal the API has to mark the job Error before it stalls
        waiting on a callback."""
        worker = self._import_worker()
        ack = worker.enqueue(
            {
                "job_id": "00000000-0000-0000-0000-000000000002",
                "preset": "codec-gs-mixed",
                "blob_url": "https://blob.example/bicycle.ply",
                "callback_url": "https://api.example/v1/jobs/y/result",
            }
        )
        self.assertFalse(ack["queued"])
        self.assertIsNotNone(ack["error"])
        self.assertIn("codec-gs-mixed", ack["error"])
        self.assertIn("SPLATFORGE_CODEC_GS_MIXED_URL", ack["error"])

    def test_fcgs_instant_without_url_returns_clear_error(self) -> None:
        worker = self._import_worker()
        ack = worker.enqueue(
            {
                "job_id": "00000000-0000-0000-0000-000000000003",
                "preset": "fcgs-instant",
                "blob_url": "https://blob.example/bonsai.ply",
                "callback_url": "https://api.example/v1/jobs/z/result",
            }
        )
        self.assertFalse(ack["queued"])
        self.assertIsNotNone(ack["error"])
        self.assertIn("fcgs-instant", ack["error"])
        self.assertIn("SPLATFORGE_FCGS_URL", ack["error"])

    def test_capture_and_compress_without_url_returns_clear_error(self) -> None:
        """`capture-and-compress` is the photos → COLMAP → 3DGS pipeline.
        Until the private `splatforge-capture` Modal app is deployed and
        its URL is wired into `SPLATFORGE_CAPTURE_URL`, the worker MUST
        surface a synchronous error naming the env var — same contract
        as the other forwarded presets."""
        worker = self._import_worker()
        ack = worker.enqueue(
            {
                "job_id": "00000000-0000-0000-0000-00000000000c",
                "preset": "capture-and-compress",
                "blob_url": "https://blob.example/photos.zip",
                "callback_url": "https://api.example/v1/jobs/c/result",
                "filename": "photos.zip",
            }
        )
        self.assertFalse(ack["queued"])
        self.assertIsNotNone(ack["error"])
        self.assertIn("capture-and-compress", ack["error"])
        self.assertIn("SPLATFORGE_CAPTURE_URL", ack["error"])

    def test_capture_and_compress_forwards_to_configured_url(self) -> None:
        """Happy path for the photos pipeline: when `SPLATFORGE_CAPTURE_URL`
        is set, the enqueue payload (including `photos.zip` filename and
        the API callback URL) is forwarded verbatim to the private app."""
        os.environ["SPLATFORGE_CAPTURE_URL"] = (
            "https://example--splatforge-capture-enqueue.modal.run"
        )
        worker = self._import_worker()

        fake_response = mock.MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"queued": True, "error": None}
        with mock.patch("requests.post", return_value=fake_response) as post:
            ack = worker.enqueue(
                {
                    "job_id": "00000000-0000-0000-0000-00000000000d",
                    "preset": "capture-and-compress",
                    "blob_url": "https://blob.example/photos.zip",
                    "callback_url": "https://api.example/v1/jobs/p/result",
                    "filename": "photos.zip",
                }
            )

        self.assertTrue(ack["queued"])
        self.assertIsNone(ack["error"])
        post.assert_called_once()
        self.assertEqual(
            post.call_args.args[0],
            "https://example--splatforge-capture-enqueue.modal.run",
        )
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["preset"], "capture-and-compress")
        self.assertEqual(body["filename"], "photos.zip")
        self.assertEqual(
            body["callback_url"], "https://api.example/v1/jobs/p/result"
        )

    def test_codec_gs_mixed_forwards_payload_to_configured_url(self) -> None:
        """Happy path: the configured Modal app gets the original payload
        verbatim (same callback_url so it reports back to the API
        directly, bypassing the public worker on the data path)."""
        os.environ["SPLATFORGE_CODEC_GS_MIXED_URL"] = (
            "https://example--splatforge-codec-gs-mixed-enqueue.modal.run"
        )
        worker = self._import_worker()

        fake_response = mock.MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"queued": True, "error": None}

        # The forwarder imports `requests` locally (so the module loads
        # without it installed); patch on the module path it'll resolve.
        with mock.patch("requests.post", return_value=fake_response) as post:
            ack = worker.enqueue(
                {
                    "job_id": "00000000-0000-0000-0000-000000000004",
                    "preset": "codec-gs-mixed",
                    "blob_url": "https://blob.example/bicycle.ply",
                    "callback_url": "https://api.example/v1/jobs/q/result",
                    "filename": "bicycle.ply",
                }
            )

        self.assertTrue(ack["queued"])
        self.assertIsNone(ack["error"])

        # Forwarded with the originating callback_url unchanged — that's
        # the contract the private Modal app relies on to call back into
        # the API directly.
        post.assert_called_once()
        args, kwargs = post.call_args
        self.assertEqual(
            args[0],
            "https://example--splatforge-codec-gs-mixed-enqueue.modal.run",
        )
        body = kwargs["json"]
        self.assertEqual(body["preset"], "codec-gs-mixed")
        self.assertEqual(
            body["callback_url"], "https://api.example/v1/jobs/q/result"
        )
        self.assertEqual(body["blob_url"], "https://blob.example/bicycle.ply")
        self.assertEqual(body["filename"], "bicycle.ply")

    def test_fcgs_instant_forwards_to_configured_url(self) -> None:
        os.environ["SPLATFORGE_FCGS_URL"] = (
            "https://example--splatforge-fcgs-enqueue.modal.run"
        )
        worker = self._import_worker()

        fake_response = mock.MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"queued": True, "error": None}
        with mock.patch("requests.post", return_value=fake_response) as post:
            ack = worker.enqueue(
                {
                    "job_id": "00000000-0000-0000-0000-000000000005",
                    "preset": "fcgs-instant",
                    "blob_url": "https://blob.example/bonsai.ply",
                    "callback_url": "https://api.example/v1/jobs/f/result",
                }
            )
        self.assertTrue(ack["queued"])
        post.assert_called_once()
        self.assertEqual(
            post.call_args.args[0],
            "https://example--splatforge-fcgs-enqueue.modal.run",
        )
        self.assertEqual(post.call_args.kwargs["json"]["preset"], "fcgs-instant")

    def test_forward_propagates_remote_error(self) -> None:
        """If the private app rejects the enqueue (e.g. its own URL is
        wired wrong or the bitstream slot is full), the public worker
        must surface that error so the API sees a non-queued ack."""
        os.environ["SPLATFORGE_CODEC_GS_MIXED_URL"] = "https://stub/"
        worker = self._import_worker()

        fake_response = mock.MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {
            "queued": False,
            "error": "missing fields: ['callback_url']",
        }
        with mock.patch("requests.post", return_value=fake_response):
            ack = worker.enqueue(
                {
                    "job_id": "00000000-0000-0000-0000-000000000006",
                    "preset": "codec-gs-mixed",
                    "blob_url": "https://blob.example/x.ply",
                    "callback_url": "https://api.example/v1/jobs/r/result",
                }
            )
        self.assertFalse(ack["queued"])
        self.assertIn("missing fields", ack["error"])

    def test_forward_handles_5xx_from_remote(self) -> None:
        """An HTTP 5xx from the private app must NOT raise — instead the
        ack carries `queued=False` + a usable error string so the API
        path that handles ModalError surfaces a 502 to the client."""
        os.environ["SPLATFORGE_FCGS_URL"] = "https://stub/"
        worker = self._import_worker()

        fake_response = mock.MagicMock()
        fake_response.status_code = 503
        fake_response.text = "upstream A100 busy"
        with mock.patch("requests.post", return_value=fake_response):
            ack = worker.enqueue(
                {
                    "job_id": "00000000-0000-0000-0000-000000000007",
                    "preset": "fcgs-instant",
                    "blob_url": "https://blob.example/x.ply",
                    "callback_url": "https://api.example/v1/jobs/s/result",
                }
            )
        self.assertFalse(ack["queued"])
        self.assertIn("503", ack["error"])
        self.assertIn("upstream A100 busy", ack["error"])

    def test_forward_handles_network_failure(self) -> None:
        """Network-level failures (DNS, TCP, timeout) collapse into a
        clean `queued=False` ack — the API never sees a bare exception
        bubble out of /enqueue."""
        os.environ["SPLATFORGE_FCGS_URL"] = "https://stub/"
        worker = self._import_worker()
        with mock.patch("requests.post", side_effect=RuntimeError("dns broke")):
            ack = worker.enqueue(
                {
                    "job_id": "00000000-0000-0000-0000-000000000008",
                    "preset": "fcgs-instant",
                    "blob_url": "https://blob.example/x.ply",
                    "callback_url": "https://api.example/v1/jobs/t/result",
                }
            )
        self.assertFalse(ack["queued"])
        self.assertIn("dns broke", ack["error"])

    def test_codec_gs_mixed_k5_shares_codec_gs_mixed_url(self) -> None:
        """`codec-gs-mixed-k5` is the same encoder as `codec-gs-mixed`
        with a different K parameter — they MUST resolve to the same
        Modal endpoint so we don't double-deploy the same container."""
        os.environ["SPLATFORGE_CODEC_GS_MIXED_URL"] = "https://shared/"
        worker = self._import_worker()

        fake_response = mock.MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"queued": True, "error": None}
        with mock.patch("requests.post", return_value=fake_response) as post:
            ack = worker.enqueue(
                {
                    "job_id": "00000000-0000-0000-0000-000000000009",
                    "preset": "codec-gs-mixed-k5",
                    "blob_url": "https://blob.example/x.ply",
                    "callback_url": "https://api.example/v1/jobs/u/result",
                }
            )
        self.assertTrue(ack["queued"])
        self.assertEqual(post.call_args.args[0], "https://shared/")
        self.assertEqual(
            post.call_args.kwargs["json"]["preset"], "codec-gs-mixed-k5"
        )

    def test_hacpp_lzma_without_url_returns_clear_error(self) -> None:
        """`hacpp-lzma` is the HAC++ Phase A + lzma anchor-feature codec
        for Scaffold-GS scenes. Until SPLATFORGE_HACPP_LZMA_URL is set,
        the worker MUST surface a synchronous error naming the env var —
        same configured-gap contract as the other forwarded presets."""
        worker = self._import_worker()
        ack = worker.enqueue(
            {
                "job_id": "00000000-0000-0000-0000-00000000000h",
                "preset": "hacpp-lzma",
                "blob_url": "https://blob.example/scaffold-bundle.tar",
                "callback_url": "https://api.example/v1/jobs/h/result",
                "filename": "scaffold-bundle.tar",
            }
        )
        self.assertFalse(ack["queued"])
        self.assertIsNotNone(ack["error"])
        self.assertIn("hacpp-lzma", ack["error"])
        self.assertIn("SPLATFORGE_HACPP_LZMA_URL", ack["error"])

    def test_hacpp_lzma_forwards_to_configured_url(self) -> None:
        """Happy path for the hacpp-lzma dispatch: the Scaffold-GS bundle
        URL + callback are forwarded verbatim. The encoder app extracts
        the tarball, runs HAC++ Phase A encode, and POSTs the terminal
        result directly to the API — the public worker stays out of the
        data path."""
        os.environ["SPLATFORGE_HACPP_LZMA_URL"] = (
            "https://example--splatforge-hacpp-lzma-enqueue.modal.run"
        )
        worker = self._import_worker()

        fake_response = mock.MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"queued": True, "error": None}
        with mock.patch("requests.post", return_value=fake_response) as post:
            ack = worker.enqueue(
                {
                    "job_id": "00000000-0000-0000-0000-00000000000a",
                    "preset": "hacpp-lzma",
                    "blob_url": "https://blob.example/bonsai-scaffold.tar",
                    "callback_url": "https://api.example/v1/jobs/h/result",
                    "filename": "bonsai-scaffold.tar",
                }
            )
        self.assertTrue(ack["queued"])
        self.assertIsNone(ack["error"])
        post.assert_called_once()
        self.assertEqual(
            post.call_args.args[0],
            "https://example--splatforge-hacpp-lzma-enqueue.modal.run",
        )
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["preset"], "hacpp-lzma")
        self.assertEqual(body["filename"], "bonsai-scaffold.tar")
        self.assertEqual(
            body["callback_url"], "https://api.example/v1/jobs/h/result"
        )

    def test_healthz_reports_dispatch_table(self) -> None:
        """Operators rely on `healthz` to confirm which preset endpoints
        are wired on a given deploy. Verify the bool-per-preset shape so
        a future refactor doesn't silently break the deploy check."""
        os.environ["SPLATFORGE_FCGS_URL"] = "https://configured/"
        os.environ["SPLATFORGE_HACPP_LZMA_URL"] = "https://configured-hacpp/"
        os.environ["SPLATFORGE_HOSTED_NEURAL_URL"] = (
            "https://configured-hosted-neural/"
        )
        worker = self._import_worker()
        body = worker.healthz()
        self.assertIn("preset_dispatch_configured", body)
        flags = body["preset_dispatch_configured"]
        self.assertTrue(flags["fcgs-instant"])
        self.assertFalse(flags["codec-gs-mixed"])
        self.assertFalse(flags["codec-gs-mixed-k5"])
        # capture-and-compress is part of the dispatch table from inception;
        # operators rely on this flag to confirm the photos pipeline is
        # wired before announcing the endpoint to design partners.
        self.assertFalse(flags["capture-and-compress"])
        # hacpp-lzma joined the dispatch table 2026-05-15. Same flag
        # contract: True when SPLATFORGE_HACPP_LZMA_URL is plumbed.
        self.assertTrue(flags["hacpp-lzma"])
        # hosted-neural joined the dispatch table 2026-05-15 (Bet 1 M3
        # productization). Per-scene A100 fit at request time; same flag
        # contract as the other presets.
        self.assertTrue(flags["hosted-neural"])
        # splatforge-qat-scaffold joined 2026-05-16 — same configured-gap
        # contract. False here because SPLATFORGE_QAT_SCAFFOLD_URL is not
        # set in this test fixture; ensures the preset surfaces as
        # unconfigured rather than silently missing from the healthz
        # report.
        self.assertFalse(flags["splatforge-qat-scaffold"])

    def test_hosted_neural_without_url_returns_clear_error(self) -> None:
        """`hosted-neural` is the per-scene A100 codec from Bet 1 / M3.
        Until SPLATFORGE_HOSTED_NEURAL_URL is set, the worker MUST surface
        a synchronous error naming the env var — same configured-gap
        contract as the other forwarded presets."""
        worker = self._import_worker()
        ack = worker.enqueue(
            {
                "job_id": "00000000-0000-0000-0000-00000000000n",
                "preset": "hosted-neural",
                "blob_url": "https://blob.example/bicycle-bundle.tar",
                "callback_url": "https://api.example/v1/jobs/n/result",
                "filename": "bicycle-bundle.tar",
            }
        )
        self.assertFalse(ack["queued"])
        self.assertIsNotNone(ack["error"])
        self.assertIn("hosted-neural", ack["error"])
        self.assertIn("SPLATFORGE_HOSTED_NEURAL_URL", ack["error"])

    def test_hosted_neural_forwards_to_configured_url(self) -> None:
        """Happy path for the hosted-neural dispatch: the bundle URL +
        callback are forwarded verbatim. The encoder app stages the
        bundle (or pulls a registered Mip-NeRF 360 scene if `filename`
        names one), runs the ~120 s per-scene neural fit on an A100,
        and POSTs the terminal result directly to the API."""
        os.environ["SPLATFORGE_HOSTED_NEURAL_URL"] = (
            "https://example--splatforge-hosted-neural-enqueue.modal.run"
        )
        worker = self._import_worker()

        fake_response = mock.MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"queued": True, "error": None}
        with mock.patch("requests.post", return_value=fake_response) as post:
            ack = worker.enqueue(
                {
                    "job_id": "00000000-0000-0000-0000-00000000000n",
                    "preset": "hosted-neural",
                    "blob_url": "https://blob.example/bicycle-bundle.tar",
                    "callback_url": "https://api.example/v1/jobs/n/result",
                    "filename": "bicycle-bundle.tar",
                }
            )
        self.assertTrue(ack["queued"])
        self.assertIsNone(ack["error"])
        post.assert_called_once()
        self.assertEqual(
            post.call_args.args[0],
            "https://example--splatforge-hosted-neural-enqueue.modal.run",
        )
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["preset"], "hosted-neural")
        self.assertEqual(body["filename"], "bicycle-bundle.tar")
        self.assertEqual(
            body["callback_url"], "https://api.example/v1/jobs/n/result"
        )


    def test_qat_scaffold_without_url_returns_clear_error(self) -> None:
        """`splatforge-qat-scaffold` is the Scaffold-GS QAT codec (2026-05-16).
        Until SPLATFORGE_QAT_SCAFFOLD_URL is set, the worker MUST surface
        a synchronous error naming the env var — same configured-gap
        contract as the other forwarded presets."""
        worker = self._import_worker()
        ack = worker.enqueue(
            {
                "job_id": "00000000-0000-0000-0000-00000000000q",
                "preset": "splatforge-qat-scaffold",
                "blob_url": "https://blob.example/bonsai-scaffold.ply",
                "callback_url": "https://api.example/v1/jobs/q/result",
                "filename": "bonsai-scaffold.ply",
            }
        )
        self.assertFalse(ack["queued"])
        self.assertIsNotNone(ack["error"])
        self.assertIn("splatforge-qat-scaffold", ack["error"])
        self.assertIn("SPLATFORGE_QAT_SCAFFOLD_URL", ack["error"])

    def test_qat_scaffold_forwards_to_configured_url(self) -> None:
        """Happy path for the qat-scaffold dispatch: the Scaffold PLY URL
        + callback are forwarded verbatim. The encoder app downloads the
        PLY, runs the quant-aware retrain on an A100, packs the
        quantized streams into a compressed .ply, uploads it, and POSTs
        the terminal result directly to the API."""
        os.environ["SPLATFORGE_QAT_SCAFFOLD_URL"] = (
            "https://example--splatforge-qat-scaffold-enqueue.modal.run"
        )
        worker = self._import_worker()

        fake_response = mock.MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"queued": True, "error": None}
        with mock.patch("requests.post", return_value=fake_response) as post:
            ack = worker.enqueue(
                {
                    "job_id": "00000000-0000-0000-0000-00000000000q",
                    "preset": "splatforge-qat-scaffold",
                    "blob_url": "https://blob.example/bonsai-scaffold.ply",
                    "callback_url": "https://api.example/v1/jobs/q/result",
                    "filename": "bonsai-scaffold.ply",
                }
            )
        self.assertTrue(ack["queued"])
        self.assertIsNone(ack["error"])
        post.assert_called_once()
        self.assertEqual(
            post.call_args.args[0],
            "https://example--splatforge-qat-scaffold-enqueue.modal.run",
        )
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["preset"], "splatforge-qat-scaffold")
        self.assertEqual(body["filename"], "bonsai-scaffold.ply")
        self.assertEqual(
            body["callback_url"], "https://api.example/v1/jobs/q/result"
        )


    def test_qat_bundle_without_url_returns_clear_error(self) -> None:
        """`splatforge-qat-bundle` is the premium-tier full-QAT recipe.
        Until SPLATFORGE_QAT_BUNDLE_URL is set, the worker MUST surface
        a synchronous error naming the env var — same configured-gap
        contract as the other forwarded presets. New 2026-05-16: this
        preset bills at $0.50/scene at Modal A100 pass-through; if it
        ever silently fell through to the local CLI we'd be running a
        ~10-minute placeholder optimize for free."""
        worker = self._import_worker()
        ack = worker.enqueue(
            {
                "job_id": "00000000-0000-0000-0000-00000000000b",
                "preset": "splatforge-qat-bundle",
                "blob_url": "https://blob.example/bundle.tar",
                "callback_url": "https://api.example/v1/jobs/b/result",
                "filename": "scaffold-bundle.tar",
            }
        )
        self.assertFalse(ack["queued"])
        self.assertIsNotNone(ack["error"])
        self.assertIn("splatforge-qat-bundle", ack["error"])
        self.assertIn("SPLATFORGE_QAT_BUNDLE_URL", ack["error"])

    def test_qat_bundle_forwards_to_configured_url(self) -> None:
        """Happy path for the premium QAT-Bundle dispatch. The bundle URL
        + callback are forwarded verbatim to the private Modal app, which
        owns the int8 retrain + constant-strip pipeline."""
        os.environ["SPLATFORGE_QAT_BUNDLE_URL"] = (
            "https://example--splatforge-qat-bundle-enqueue.modal.run"
        )
        worker = self._import_worker()

        fake_response = mock.MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"queued": True, "error": None}
        with mock.patch("requests.post", return_value=fake_response) as post:
            ack = worker.enqueue(
                {
                    "job_id": "00000000-0000-0000-0000-00000000000B",
                    "preset": "splatforge-qat-bundle",
                    "blob_url": "https://blob.example/bonsai-bundle.tar",
                    "callback_url": "https://api.example/v1/jobs/B/result",
                    "filename": "bonsai-bundle.tar",
                }
            )
        self.assertTrue(ack["queued"])
        self.assertIsNone(ack["error"])
        post.assert_called_once()
        self.assertEqual(
            post.call_args.args[0],
            "https://example--splatforge-qat-bundle-enqueue.modal.run",
        )
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["preset"], "splatforge-qat-bundle")
        self.assertEqual(body["filename"], "bonsai-bundle.tar")

    def test_healthz_reports_qat_bundle_flag(self) -> None:
        """Operators rely on healthz to confirm the premium preset is
        wired. QAT-Bundle was added 2026-05-16; ensure it appears in
        the preset_dispatch_configured map so the deploy check covers
        the new preset."""
        worker = self._import_worker()
        body = worker.healthz()
        flags = body["preset_dispatch_configured"]
        self.assertIn(
            "splatforge-qat-bundle",
            flags,
            "qat-bundle must appear in healthz dispatch table",
        )
        # capture-and-compress shipped earlier; pin that it's also surfaced.
        self.assertIn("capture-and-compress", flags)

    def test_required_keys_covers_every_dispatch_env_var(self) -> None:
        """The Modal Secret `required_keys` list MUST name every env var
        that PRESET_DISPATCH_URLS reads. If a new preset is added to the
        dispatch table but its env var is left out of required_keys, the
        Modal app launches without that secret bound — the operator only
        finds out when a customer hits the preset and gets a generic
        500. Regression: SPLATFORGE_CAPTURE_URL was missing from
        required_keys when the photos pipeline shipped."""
        # Parse the worker.py source directly so the test catches
        # additions to PRESET_DISPATCH_URLS that forget to update
        # the Secret declaration.
        import re as _re
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "worker.py"), "r", encoding="utf-8") as fh:
            src = fh.read()

        # Collect env-var names referenced inside PRESET_DISPATCH_URLS.
        dispatch_block_start = src.index("PRESET_DISPATCH_URLS = {")
        dispatch_block_end = src.index("\n}\n", dispatch_block_start)
        dispatch_block = src[dispatch_block_start:dispatch_block_end]
        dispatch_envs = set(_re.findall(r"SPLATFORGE_[A-Z0-9_]+_URL", dispatch_block))

        # Collect env-var names declared in the asgi_app Secret.from_name
        # required_keys list. Find the FIRST `required_keys=[` after the
        # WORKER_ASGI_LABEL block to avoid colliding with run_optimize's
        # vercel-blob Secret.
        asgi_block_start = src.index('@modal.asgi_app(label=WORKER_ASGI_LABEL)')
        # Walk back to find the function block's secrets= clause.
        web_app_block_start = src.rindex("def web_app", 0, asgi_block_start + 200)
        function_block_start = src.rindex("@app.function", 0, web_app_block_start)
        # Find required_keys=[...] inside this function block.
        m = _re.search(
            r'required_keys=\[(.*?)\]',
            src[function_block_start:asgi_block_start + 200],
            flags=_re.DOTALL,
        )
        self.assertIsNotNone(
            m, "could not find required_keys list on the web_app Secret"
        )
        declared = set(_re.findall(r'"(SPLATFORGE_[A-Z0-9_]+_URL)"', m.group(1)))

        missing = dispatch_envs - declared
        self.assertFalse(
            missing,
            f"PRESET_DISPATCH_URLS references env vars that are NOT in "
            f"the web_app Modal Secret required_keys list: {sorted(missing)}",
        )

    def test_capture_url_in_required_keys(self) -> None:
        """Explicit regression for the SPLATFORGE_CAPTURE_URL omission.
        capture-and-compress was added to PRESET_DISPATCH_URLS but its
        env var was missing from required_keys until 2026-05-16. Pin
        the fix so a refactor can't undo it silently."""
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "worker.py"), "r", encoding="utf-8") as fh:
            src = fh.read()
        # Find the asgi_app Secret block and check the literal is present.
        asgi_block_start = src.index('@modal.asgi_app(label=WORKER_ASGI_LABEL)')
        function_block_start = src.rindex("@app.function", 0, asgi_block_start)
        block = src[function_block_start:asgi_block_start + 500]
        self.assertIn("SPLATFORGE_CAPTURE_URL", block)

if __name__ == "__main__":
    unittest.main()
