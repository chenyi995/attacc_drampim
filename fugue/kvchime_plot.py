"""Compatibility wrapper for current single-metric publication packages.

Requires the full all-models run (both sweep and workloads).
"""
import runpy
runpy.run_module('fugue.kvchime_paper',run_name='__main__')
