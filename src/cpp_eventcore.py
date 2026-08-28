"""ctypes bridge to the native event-schedule core (libeventcore.so).

Experimental branch chenyi-822-cppcore-exp (chenyi9 2026-08-28).  The core
mirrors every DAG event as (device id, duration, dep indices) and owns the
scheduling state machine; Python keeps all domain logic and the SplitEvent
records.  Disable with KVPIM_CPPCORE=0 (pure-Python paths are preserved
verbatim and remain the cross-check: the overlap-contract validator replays
the native schedule in Python every run).
"""
from __future__ import annotations

import ctypes
import os
from array import array
from typing import Dict, Optional, Sequence

_LIB_PATH = os.path.join(os.path.dirname(__file__), "cppcore", "libeventcore.so")


def _load_library() -> Optional[ctypes.CDLL]:
    if os.environ.get("KVPIM_CPPCORE", "1") == "0":
        return None
    try:
        lib = ctypes.CDLL(_LIB_PATH)
    except OSError:
        return None
    lib.ec_new.restype = ctypes.c_void_p
    lib.ec_new.argtypes = [ctypes.c_int]
    lib.ec_free.argtypes = [ctypes.c_void_p]
    lib.ec_size.restype = ctypes.c_int64
    lib.ec_size.argtypes = [ctypes.c_void_p]
    lib.ec_add.restype = ctypes.c_int64
    lib.ec_add.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_double,
                           ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
    lib.ec_set_duration.argtypes = [ctypes.c_void_p, ctypes.c_int64,
                                    ctypes.c_double]
    lib.ec_reset.argtypes = [ctypes.c_void_p]
    lib.ec_advance.restype = ctypes.c_int64
    lib.ec_advance.argtypes = [ctypes.c_void_p]
    lib.ec_end.restype = ctypes.c_double
    lib.ec_end.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    lib.ec_bulk_times.argtypes = [ctypes.c_void_p,
                                  ctypes.POINTER(ctypes.c_double),
                                  ctypes.POINTER(ctypes.c_double)]
    return lib


_LIB = _load_library()


def event_index(event_id: str) -> int:
    """Both id schemes ("cb-N", "legacy-N") encode the list index."""
    return int(event_id.rsplit("-", 1)[1])


class EventCore:
    """One run's native schedule state (create per report run)."""

    def __init__(self, pipe: bool):
        if _LIB is None:
            raise RuntimeError("eventcore library unavailable")
        self._lib = _LIB
        self._handle = _LIB.ec_new(1 if pipe else 0)
        self._device_ids: Dict[str, int] = {}

    def close(self) -> None:
        if self._handle:
            self._lib.ec_free(self._handle)
            self._handle = None

    def _device(self, device: str) -> int:
        did = self._device_ids.get(device)
        if did is None:
            did = len(self._device_ids)
            self._device_ids[device] = did
        return did

    def add(self, device: str, duration: float,
            dep_ids: Sequence[str]) -> None:
        ndep = len(dep_ids)
        buf = (ctypes.c_int32 * ndep)(*[event_index(d) for d in dep_ids]) \
            if ndep else None
        index = self._lib.ec_add(self._handle, self._device(device),
                                 float(duration), buf, ndep)
        if index < 0:
            raise ValueError("eventcore: dependency on a future event")

    def set_duration(self, index: int, duration: float) -> None:
        self._lib.ec_set_duration(self._handle, index, float(duration))

    def reset(self) -> None:
        self._lib.ec_reset(self._handle)

    def advance(self) -> None:
        self._lib.ec_advance(self._handle)

    def end(self, event_id: str) -> float:
        return self._lib.ec_end(self._handle, event_index(event_id))

    def size(self) -> int:
        return int(self._lib.ec_size(self._handle))

    def bulk_times(self, count: int):
        start = array("d", bytes(8 * count))
        end = array("d", bytes(8 * count))
        self._lib.ec_bulk_times(
            self._handle,
            ctypes.cast(start.buffer_info()[0], ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(end.buffer_info()[0], ctypes.POINTER(ctypes.c_double)))
        return start, end


def new_core(pipe: bool) -> Optional[EventCore]:
    if _LIB is None:
        return None
    return EventCore(pipe)
