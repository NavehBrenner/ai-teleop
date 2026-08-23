"""Tests for the project console logger (LAB-48).

Covers the module's contract: child-logger naming, idempotent configuration,
level filtering, the optional file tee, and the rich-absent fallback. Output
styling (rich vs stdlib) is selected from ``RichHandler`` availability + TTY;
here we exercise the selection logic, not rich's rendering.
"""

from __future__ import annotations

import argparse
import io
import logging

import pytest

from ai_teleop.common import log as logmod
from ai_teleop.common.log import (
    PACKAGE_LOGGER_NAME,
    configure_logging,
    get_logger,
)


@pytest.fixture(autouse=True)
def _reset_package_logger():
    """Strip handlers off the shared package logger around each test."""
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    saved_level = logger.level
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    yield
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(saved_level)


def test_get_logger_is_namespaced_child():
    logger = get_logger("datagen")
    assert logger.name == f"{PACKAGE_LOGGER_NAME}.datagen"
    assert logger.parent is logging.getLogger(PACKAGE_LOGGER_NAME)


def test_configure_logging_is_idempotent():
    configure_logging(use_rich=False)
    configure_logging(use_rich=False)
    configure_logging(use_rich=False)
    handlers = logging.getLogger(PACKAGE_LOGGER_NAME).handlers
    # One console handler, no matter how many times we (re)configure.
    assert len(handlers) == 1


def test_level_filtering_suppresses_below_threshold(caplog):
    # No caplog.at_level() here: it would override the very level we're testing.
    # configure_logging sets the package logger to WARNING, so the INFO record is
    # filtered at the logger and never reaches caplog's (propagated) handler.
    configure_logging(level="WARNING", use_rich=False)
    logger = get_logger("leveltest")
    logger.info("hidden-info")
    logger.warning("shown-warning")
    messages = [record.message for record in caplog.records]
    assert "shown-warning" in messages
    assert "hidden-info" not in messages


def test_log_file_tee_writes_records(tmp_path):
    log_path = tmp_path / "run.log"
    configure_logging(level="INFO", log_file=log_path, use_rich=False)
    get_logger("filetest").info("written to %s", "disk")
    assert log_path.exists()
    contents = log_path.read_text(encoding="utf-8")
    assert "written to disk" in contents
    assert "[filetest]" in contents  # short-name tag, not the ai_teleop. prefix


def test_falls_back_to_stdlib_when_rich_absent(monkeypatch):
    # rich unavailable: must not pick RichHandler, must not raise — even with
    # use_rich=True and a (faked) TTY, the `RichHandler is not None` guard holds.
    monkeypatch.setattr(logmod, "RichHandler", None)
    monkeypatch.setattr(logmod.sys, "stderr", _FakeTTY())
    configure_logging(use_rich=True)
    (handler,) = logging.getLogger(PACKAGE_LOGGER_NAME).handlers
    assert isinstance(handler, logging.StreamHandler)


class _FakeTTY:
    """Minimal writable, TTY-reporting stand-in for sys.stderr."""

    def isatty(self) -> bool:
        return True

    def write(self, _text: str) -> int:
        return 0

    def flush(self) -> None:
        pass


def test_unknown_level_raises():
    with pytest.raises(ValueError, match="unknown log level"):
        configure_logging(level="LOUD")


# --------------------------------------------------------------------------
# Console encoding (LAB-125) — `--help` must not die on a legacy code page
# --------------------------------------------------------------------------


def _cp1252_stream() -> tuple[io.TextIOWrapper, io.BytesIO]:
    """A strict cp1252 console, the way Windows hands one to a Python process.

    Returns the raw ``BytesIO`` alongside it: reaching it back through
    ``TextIOWrapper.buffer`` is untyped, and the bytes are what the assertions check.
    """
    raw = io.BytesIO()
    return io.TextIOWrapper(raw, encoding="cp1252", errors="strict"), raw


def test_cp1252_console_rejects_the_help_text_before_the_fix():
    """Pins the failure being fixed, so the regression test below has teeth.

    ``Δ`` (U+0394) appears in ``run_episode.py``'s ``--max-dpos``/``--policy`` help, and
    argparse writes ``--help`` to ``sys.stdout`` in strict mode.
    """
    stream, _ = _cp1252_stream()
    with pytest.raises(UnicodeEncodeError):
        stream.write("per-step Δ bound")
        stream.flush()


def test_ensure_console_encoding_makes_non_ascii_writable(monkeypatch):
    """After the fix the same write succeeds, on stdout *and* stderr."""
    out, out_raw = _cp1252_stream()
    err, err_raw = _cp1252_stream()
    monkeypatch.setattr(logmod.sys, "stdout", out)
    monkeypatch.setattr(logmod.sys, "stderr", err)

    logmod.ensure_console_encoding()

    for stream, raw in ((out, out_raw), (err, err_raw)):
        stream.write("per-step Δ bound")
        stream.flush()
        assert "Δ".encode() in raw.getvalue()  # re-encoded as real UTF-8


def test_ensure_console_encoding_survives_streams_it_cannot_touch(monkeypatch):
    """Never the reason a command fails: unreconfigurable streams are left alone."""

    class _NoReconfigure:
        def write(self, _text: str) -> int:
            return 0

    monkeypatch.setattr(logmod.sys, "stdout", _NoReconfigure())
    monkeypatch.setattr(logmod.sys, "stderr", io.StringIO())
    logmod.ensure_console_encoding()  # must not raise


def test_argparse_help_renders_over_a_cp1252_console(monkeypatch):
    """The end-to-end shape of the bug: build a parser, print its help, no crash.

    ``add_logging_arguments`` is where the fix is wired in, and every entry point calls
    it while building its parser — before ``parse_args`` can print anything.
    """
    stream, raw = _cp1252_stream()
    monkeypatch.setattr(logmod.sys, "stdout", stream)

    parser = argparse.ArgumentParser(prog="demo")
    parser.add_argument("--max-dpos", help="per-step Δ bound in m/step")
    logmod.add_logging_arguments(parser)

    stream.write(parser.format_help())
    stream.flush()
    assert b"--log-level" in raw.getvalue()
