"""Abandoned publication evaluations must converge instead of staying active forever.

A publication evaluation whose Runner already produced the deliverable can stay in state
"running" when the Runner goes offline before the Hub observes the result. The row then blocks
the publication gate until somebody cancels it by hand.
"""
import os
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

# Set this before any app import so the test never inherits a developer database.
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

from app.services import plugin_evaluations as evaluations


def _row(state: str = "running", age_seconds: int = 0, started: bool = True):
    """Build a minimal evaluation run double."""
    started_at = None
    if started:
        started_at = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    return SimpleNamespace(
        id="run-1", plugin_id="plugin-1", version_id="version-1", case_key="sales-analysis",
        org_id=None, state=state, verdict="pending", external_ref="ref-1", active_key="active-1",
        agent_kind="local:pi", model="demo-model", case_snapshot={}, executor={}, result={},
        error="", created_by="tester", reviewed_by="", review_note="",
        started_at=started_at, completed_at=None, reviewed_at=None,
    )


class _Rows:
    """Minimal stand-in for a SQLAlchemy result."""

    def __init__(self, rows):
        self._rows = list(rows)

    def scalars(self):
        return self

    def all(self):
        return list(self._rows)


class TimeoutBudgetTests(unittest.TestCase):
    def test_expired_row_is_recognized(self):
        self.assertTrue(evaluations._expired(_row(age_seconds=3 * 3600)))
        self.assertFalse(evaluations._expired(_row(age_seconds=60)))

    def test_naive_started_at_is_read_as_utc(self):
        row = _row()
        row.started_at = (datetime.now(timezone.utc) - timedelta(hours=3)).replace(tzinfo=None)
        self.assertTrue(evaluations._expired(row))

    def test_row_without_start_time_never_expires(self):
        self.assertFalse(evaluations._expired(_row(started=False)))


class RefreshConvergenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_unreachable_runner_fails_an_expired_evaluation(self):
        row = _row(age_seconds=3 * 3600)
        session = SimpleNamespace(commit=AsyncMock())
        with patch.object(evaluations.ai_exec, "load", AsyncMock(return_value={})), \
                patch.object(evaluations.ai_exec, "status_sync",
                             Mock(return_value={"ok": False, "error": "本地会话不在了"})):
            await evaluations.refresh(session, row)
        self.assertEqual(row.state, "failed")
        self.assertIsNotNone(row.completed_at)
        self.assertIsNone(row.active_key)
        self.assertIn("超时", row.error)
        # The last probe stays available as evidence for the failed attempt.
        self.assertFalse(row.result["lastProbe"]["ok"])

    async def test_unreachable_runner_keeps_a_recent_evaluation_active(self):
        row = _row(age_seconds=60)
        session = SimpleNamespace(commit=AsyncMock())
        with patch.object(evaluations.ai_exec, "load", AsyncMock(return_value={})), \
                patch.object(evaluations.ai_exec, "status_sync",
                             Mock(return_value={"ok": False, "error": "本地会话不在了"})):
            await evaluations.refresh(session, row)
        self.assertEqual(row.state, "running")
        self.assertIsNone(row.completed_at)
        self.assertEqual(row.error, "")
        self.assertFalse(row.result["lastProbe"]["ok"])

    async def test_running_evaluation_fails_after_the_timeout_budget(self):
        row = _row(age_seconds=3 * 3600)
        session = SimpleNamespace(commit=AsyncMock())
        with patch.object(evaluations.ai_exec, "load", AsyncMock(return_value={})), \
                patch.object(evaluations.ai_exec, "status_sync",
                             Mock(return_value={"ok": True, "done": False, "running": True})):
            await evaluations.refresh(session, row)
        self.assertEqual(row.state, "failed")
        self.assertIsNotNone(row.completed_at)
        self.assertIn("超时", row.error)


class SweepTests(unittest.IsolatedAsyncioTestCase):
    async def test_sweep_fails_only_expired_evaluations(self):
        expired = _row(age_seconds=3 * 3600)
        fresh = _row(age_seconds=30)
        session = SimpleNamespace(commit=AsyncMock(),
                                  execute=AsyncMock(return_value=_Rows([expired, fresh])))
        out = await evaluations.sweep_once(session)
        self.assertEqual(out["active"], 2)
        self.assertEqual(out["expired"], 1)
        self.assertEqual(expired.state, "failed")
        self.assertIsNotNone(expired.completed_at)
        self.assertIsNone(expired.active_key)
        self.assertEqual(fresh.state, "running")
        self.assertIsNone(fresh.completed_at)
        session.commit.assert_awaited()

    async def test_sweep_without_expired_evaluations_does_not_commit(self):
        fresh = _row(age_seconds=30)
        session = SimpleNamespace(commit=AsyncMock(),
                                  execute=AsyncMock(return_value=_Rows([fresh])))
        out = await evaluations.sweep_once(session)
        self.assertEqual(out["expired"], 0)
        self.assertEqual(fresh.state, "running")
        session.commit.assert_not_awaited()


class SweepRegistrationTests(unittest.TestCase):
    def test_hub_supervises_the_evaluation_sweep(self):
        main = (Path(__file__).parent / "app" / "main.py").read_text(encoding="utf-8")
        self.assertIn('("plugin_evaluation_sweep", plugin_evaluations.sweep_loop)', main)


if __name__ == "__main__":
    unittest.main()
