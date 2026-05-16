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
    modal.Secret = _Secret  # type: ignore[attr-defined]
    modal.App = _App  # type: ignore[attr-defined]
    modal.fastapi_endpoint = _fastapi_endpoint  # type: ignore[attr-defined]
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


if __name__ == "__main__":
    unittest.main()
