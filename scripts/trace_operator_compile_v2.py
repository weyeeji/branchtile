#!/usr/bin/env python3
"""Trace a bounded qualification attempt without changing compiler passes.

All arguments are forwarded unchanged to qualify_static_bank_v2.py. Every 20
seconds faulthandler prints the current Python stack to stderr; SIGUSR1 also
prints a stack in this wrapper's process. No signal is sent by this script.
"""
from __future__ import annotations

import faulthandler
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import time


def main():
    assert os.geteuid() != 0, "No root execution"
    assert sys.prefix != sys.base_prefix, "Project virtual environment required"
    wrapper = Path(__file__).resolve()
    target = wrapper.with_name("qualify_static_bank_v2.py")
    arguments = tuple(sys.argv[1:])
    metadata = {
        "event": "operator_compile_trace_started",
        "wrapper_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
        "target_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "forwarded_argv_sha256": hashlib.sha256(json.dumps(arguments, ensure_ascii=False).encode()).hexdigest(),
        "forwarded_argument_count": len(arguments),
        "uid": os.geteuid(), "pid": os.getpid(), "python": sys.executable,
        "monotonic_seconds": time.monotonic(),
        "traceback_interval_seconds": 20, "repeat": True,
        "compiler_configuration_changed": False,
        "SIGUSR1_registered": hasattr(signal, "SIGUSR1"),
    }
    print(json.dumps(metadata), flush=True)
    faulthandler.enable(file=sys.stderr, all_threads=True)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    faulthandler.dump_traceback_later(20, repeat=True, file=sys.stderr, exit=False)
    try:
        # run_path changes only argv[0] to target while executing __main__;
        # argv[1:] retains every original token and its ordering.
        runpy.run_path(str(target), run_name="__main__")
        assert tuple(sys.argv[1:]) == arguments, "Forwarded CLI arguments changed"
    finally:
        faulthandler.cancel_dump_traceback_later()
        if hasattr(signal, "SIGUSR1"):
            faulthandler.unregister(signal.SIGUSR1)
        print(json.dumps({"event": "operator_compile_trace_finished",
                          "monotonic_seconds": time.monotonic()}), flush=True)


if __name__ == "__main__":
    main()
