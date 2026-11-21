"""Model download lifecycle with real temporary files and isolated Hub transfers."""

import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from broca.ai_model import AiModel
from broca.utils._download_state import DownloadManifest


class AiModelDownloadTests(unittest.IsolatedAsyncioTestCase):
    """Exercise persistent downloads without downloading model weights."""

    def setUp(self) -> None:
        """Redirect model storage to a temporary home for every test.

        :returns: None.
        """
        self.home = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(Path, "home", return_value=self.home))

    def write_asset(self, relative: str, content: bytes = b"weights") -> str:
        """Create a local asset returned by an isolated Hub transfer.

        :param relative: Path relative to the AI storage root.
        :param content: Bytes to persist.
        :returns: Absolute asset path.
        """
        path = self.home / "ai" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return str(path)

    async def test_single_checkpoint_and_projector_persist_completion(self) -> None:
        """Remember all assets and invalidate completion after asset deletion.

        :returns: None.
        """
        model = AiModel(
            id="lab/chat",
            file="nested/model.gguf",
            clip="projector.gguf",
            revision="pinned",
        )
        self.assertFalse(await model.is_downloaded())
        self.assertEqual(await model.download_progress(), 0)
        paths = [
            self.write_asset("lab/chat/nested/model.gguf"),
            self.write_asset("lab/chat/projector.gguf"),
        ]
        with patch("broca.ai_model.hf_hub_download", side_effect=paths) as transfer:
            self.assertTrue(await model.download())
            self.assertTrue(
                await AiModel.model_validate_json(model.model_dump_json()).download()
            )
        self.assertEqual(transfer.call_count, 2)
        for call in transfer.call_args_list:
            self.assertEqual(call.kwargs["local_dir"], str(model.local_path))
            self.assertEqual(call.kwargs["revision"], "pinned")
        self.assertTrue(await model.model_copy().is_downloaded())
        self.assertEqual(await model.download_progress(), 1)
        Path(paths[0]).unlink()
        self.assertFalse(await model.is_downloaded())
        self.assertEqual(await model.download_progress(), 0)

    async def test_snapshot_filters_and_offline_options(self) -> None:
        """Forward snapshot selection and record only downloaded model assets.

        :returns: None.
        """
        model = AiModel(
            id="lab/snapshot",
            allow_patterns=["*.json"],
            ignore_patterns=["large/*"],
            local_files_only=True,
            max_workers=2,
        )
        self.write_asset("lab/snapshot/config.json", b"{}")
        metadata = Path(
            self.write_asset(
                "lab/snapshot/.cache/huggingface/download/config.json.metadata"
            )
        )
        with patch(
            "broca.ai_model.snapshot_download", return_value=str(model.local_path)
        ) as transfer:
            self.assertTrue(await model.download())
        self.assertEqual(transfer.call_args.kwargs["allow_patterns"], ["*.json"])
        self.assertEqual(transfer.call_args.kwargs["ignore_patterns"], ["large/*"])
        self.assertTrue(transfer.call_args.kwargs["local_files_only"])
        self.assertEqual(transfer.call_args.kwargs["max_workers"], 2)
        metadata.unlink()
        self.assertTrue(await model.is_downloaded())

    async def test_download_through_symlinked_home(self) -> None:
        """Persist completion when the Hub returns the canonical storage path."""
        model = AiModel(id="lab/model", file="model.gguf")
        checkpoint = Path(self.write_asset("lab/model/model.gguf")).resolve()
        alias = self.home / "home-link"
        alias.symlink_to(self.home, target_is_directory=True)
        with (
            patch.object(Path, "home", return_value=alias),
            patch("broca.ai_model.hf_hub_download", return_value=str(checkpoint)),
        ):
            self.assertTrue(await model.download())
            self.assertTrue(await model.is_downloaded())

    async def test_separate_weights_download_pipeline_and_checkpoint(self) -> None:
        """Preserve the reference pack's quantized Diffusers download stages.

        :returns: None.
        """
        model = AiModel(
            id="lab/pipeline",
            weights_repo="quantized/weights",
            file="model.gguf",
            clip="projector.gguf",
            revision="pipeline-rev",
            weights_revision="weights-rev",
            ignore_patterns=["*.onnx"],
        )
        self.write_asset("lab/pipeline/model_index.json", b"{}")
        weights = self.write_asset("quantized/weights/model.gguf")
        projector = self.write_asset("lab/pipeline/projector.gguf")
        with (
            patch(
                "broca.ai_model.snapshot_download",
                return_value=str(model.local_path),
            ) as snapshot,
            patch(
                "broca.ai_model.hf_hub_download", side_effect=[weights, projector]
            ) as files,
        ):
            self.assertTrue(await model.download())
        self.assertEqual(snapshot.call_args.kwargs["repo_id"], model.id)
        self.assertIn(
            "transformer/*.safetensors", snapshot.call_args.kwargs["ignore_patterns"]
        )
        self.assertIn("*.onnx", snapshot.call_args.kwargs["ignore_patterns"])
        self.assertEqual(files.call_args_list[0].kwargs["repo_id"], model.weights_repo)
        self.assertEqual(files.call_args_list[0].kwargs["revision"], "weights-rev")
        self.assertEqual(
            files.call_args_list[0].kwargs["local_dir"],
            str(self.home / "ai/quantized/weights"),
        )
        self.assertEqual(files.call_args_list[1].kwargs["revision"], "pipeline-rev")
        self.assertTrue(await model.is_downloaded())

    async def test_failed_download_can_be_retried(self) -> None:
        """Expose transfer failures and retry without publishing false completion.

        :returns: None.
        """
        model = AiModel(id="lab/model", file="model.gguf")
        with patch("broca.ai_model.hf_hub_download", side_effect=OSError("offline")):
            with self.assertRaisesRegex(OSError, "offline"):
                await model.download()
        self.assertFalse(await model.is_downloaded())
        self.assertEqual(await model.download_progress(), 0)
        with patch(
            "broca.ai_model.hf_hub_download",
            return_value=self.write_asset("lab/model/model.gguf"),
        ):
            self.assertTrue(await model.download())

    async def test_concurrent_callers_wait_for_one_transfer_and_share_progress(
        self,
    ) -> None:
        """Share byte progress and complete both callers after the transfer ends.

        :returns: None.
        """
        model = AiModel(id="lab/model", file="model.gguf")
        started, release = threading.Event(), threading.Event()

        def transfer(**options: Any) -> str:
            """Block a byte-reporting transfer until its state has been inspected.

            :param options: Arguments normally passed to Hugging Face.
            :returns: Downloaded checkpoint path.
            """
            with options["tqdm_class"](total=100, unit="B", disable=True) as progress:
                progress.update(25)
                started.set()
                if not release.wait(5):
                    raise TimeoutError("test transfer was not released")
                progress.update(75)
            return self.write_asset("lab/model/model.gguf")

        with patch(
            "broca.ai_model.hf_hub_download", side_effect=transfer
        ) as download:
            first = asyncio.create_task(model.download())
            second: asyncio.Task[bool] | None = None
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                self.assertAlmostEqual(
                    await model.model_copy().download_progress(), 0.25
                )
                self.assertFalse(await model.is_downloaded())
                second = asyncio.create_task(model.model_copy().download())
                await asyncio.sleep(0.1)
                self.assertFalse(second.done())
                self.assertEqual(download.call_count, 1)
            finally:
                release.set()
                results = await asyncio.gather(first, *([second] if second else []))
            self.assertTrue(all(results))
            self.assertEqual(download.call_count, 1)

    async def test_repeated_cancellation_retains_transfer_lock(self) -> None:
        """Keep a cancelled transfer isolated until its worker has exited.

        :returns: None.
        """
        model = AiModel(id="lab/model", file="model.gguf")
        started, release = threading.Event(), threading.Event()

        def transfer(**options: Any) -> str:
            """Keep native I/O active across cancellation of the async caller.

            :param options: Hub transfer arguments.
            :returns: Downloaded checkpoint path.
            """
            started.set()
            if not release.wait(5):
                raise TimeoutError("test transfer was not released")
            return self.write_asset("lab/model/model.gguf")

        with patch(
            "broca.ai_model.hf_hub_download", side_effect=transfer
        ) as download:
            task = asyncio.create_task(model.download())
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                self.assertGreater(await model.download_progress(), 0)
                self.assertLess(await model.download_progress(), 1)
                self.assertEqual(download.call_count, 1)
            finally:
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        self.assertFalse(await model.is_downloaded())

    async def test_different_revisions_do_not_write_one_repository_concurrently(
        self,
    ) -> None:
        """Serialize different selections that share the same local repository.

        :returns: None.
        """
        first_model = AiModel(id="lab/model", file="model.gguf", revision="v1")
        second_model = AiModel(id="lab/model", file="model.gguf", revision="v2")
        started, release = threading.Event(), threading.Event()

        def transfer(**options: Any) -> str:
            """Keep one revision active while a second revision is requested.

            :param options: Hub transfer arguments including the selected revision.
            :returns: Downloaded checkpoint path.
            """
            if options["revision"] == "v1":
                started.set()
                if not release.wait(5):
                    raise TimeoutError("test transfer was not released")
            return self.write_asset(
                "lab/model/model.gguf", str(options["revision"]).encode()
            )

        with patch(
            "broca.ai_model.hf_hub_download", side_effect=transfer
        ) as download:
            first = asyncio.create_task(first_model.download())
            second: asyncio.Task[bool] | None = None
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                second = asyncio.create_task(second_model.download())
                await asyncio.sleep(0.1)
                self.assertEqual(download.call_count, 1)
                self.assertFalse(second.done())
            finally:
                release.set()
                results = await asyncio.gather(first, *([second] if second else []))
            self.assertTrue(all(results))
            self.assertEqual(download.call_count, 2)
        self.assertFalse(await first_model.is_downloaded())
        self.assertTrue(await second_model.is_downloaded())

    async def test_empty_snapshot_is_not_complete(self) -> None:
        """Do not treat a snapshot containing only Hub metadata as a model.

        :returns: None.
        """
        model = AiModel(id="lab/empty")
        self.write_asset("lab/empty/.cache/huggingface/download/empty.metadata")
        with patch(
            "broca.ai_model.snapshot_download", return_value=str(model.local_path)
        ):
            with self.assertRaisesRegex(OSError, "assets"):
                await model.download()
        self.assertFalse(await model.is_downloaded())

    async def test_changed_revision_or_filters_require_fresh_completion(self) -> None:
        """Keep download state specific to the selected revision and file filters.

        :returns: None.
        """
        model = AiModel(id="lab/model", file="model.gguf", revision="v1")
        with patch(
            "broca.ai_model.hf_hub_download",
            return_value=self.write_asset("lab/model/model.gguf"),
        ):
            await model.download()
        self.assertFalse(
            await AiModel(
                id="lab/model", file="model.gguf", revision="v2"
            ).is_downloaded()
        )
        changed = AiModel(
            id="lab/model", file="model.gguf", revision="v1", ignore_patterns=["*.bin"]
        )
        self.assertFalse(await changed.is_downloaded())
        model.checkpoint_path.write_bytes(b"changed checkpoint")
        self.assertFalse(await model.is_downloaded())

    async def test_corrupt_or_stale_state_is_missing(self) -> None:
        """Ignore corrupt manifests and stale progress left by interrupted processes.

        :returns: None.
        """
        model = AiModel(id="lab/model")
        model.state_path.parent.mkdir(parents=True)
        for content in (
            "broken json",
            "[]",
            '{"status": "downloading", "progress": 0.4}',
            json.dumps({"files": ["/outside"]}),
        ):
            model.state_path.write_text(content)
            self.assertFalse(await model.is_downloaded())
            self.assertEqual(await model.download_progress(), 0)

    async def test_snapshot_completion_does_not_include_filtered_local_files(
        self,
    ) -> None:
        """Track selected snapshot assets independently of preexisting local files.

        :returns: None.
        """
        model = AiModel(
            id="lab/model", allow_patterns=["*.json"], ignore_patterns=["ignored.json"]
        )
        self.write_asset("lab/model/config.json", b"{}")
        old_weights = Path(self.write_asset("lab/model/old-model.safetensors"))
        ignored = Path(self.write_asset("lab/model/ignored.json", b"{}"))
        with patch(
            "broca.ai_model.snapshot_download", return_value=str(model.local_path)
        ):
            await model.download()
        old_weights.unlink()
        ignored.unlink()
        self.assertTrue(await model.is_downloaded())

    async def test_download_progress_rechecks_completion_after_lock_acquisition(
        self,
    ) -> None:
        """Return complete if the transfer finishes between reading and locking.

        :returns: None.
        """
        model = AiModel(id="lab/model", file="model.gguf")
        with patch(
            "broca.ai_model.hf_hub_download",
            return_value=self.write_asset("lab/model/model.gguf"),
        ):
            await model.download()
        complete = model.state_path.read_bytes()
        with patch(
            "broca.ai_model.read_manifest",
            side_effect=[
                DownloadManifest(status="downloading", progress=0.5),
                DownloadManifest.model_validate_json(complete),
            ],
        ):
            self.assertEqual(await model.download_progress(), 1)

    async def test_rejects_repository_symlink_outside_model_root(self) -> None:
        """Prevent a repository symlink from redirecting model writes elsewhere.

        :returns: None.
        """
        outside = self.home / "outside"
        outside.mkdir()
        root = self.home / "ai/lab"
        root.mkdir(parents=True)
        (root / "model").symlink_to(outside, target_is_directory=True)
        with patch("broca.ai_model.hf_hub_download") as transfer:
            with self.assertRaises(ValueError):
                await AiModel(id="lab/model", file="model.gguf").download()
        transfer.assert_not_called()
