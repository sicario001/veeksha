"""Build the optional native (C++) receive engine on install.

``veeksha_native`` is a pybind11 free-threaded extension (``py::mod_gil_not_used``)
that owns the microsecond-sensitive transport + timing path. It is OPTIONAL: if a
C++ toolchain or pybind11 is unavailable, the build degrades gracefully and
``veeksha.native.native_available()`` returns False so callers fall back to the
Python transport. Metadata otherwise comes from pyproject.toml.
"""

from __future__ import annotations

from setuptools import setup
from setuptools.command.build_ext import build_ext

try:
    from pybind11.setup_helpers import Pybind11Extension

    ext_modules = [
        Pybind11Extension(
            "veeksha.native.veeksha_native",
            ["veeksha/native/src/native_receiver.cpp"],
            cxx_std=17,
        )
    ]
except Exception:  # pragma: no cover - pybind11 missing at build time
    ext_modules = []


class OptionalBuildExt(build_ext):
    """Never fail the wheel because the native extension didn't compile."""

    def run(self) -> None:
        try:
            super().run()
        except Exception as exc:  # pragma: no cover - toolchain-dependent
            self._warn(exc)

    def build_extension(self, ext) -> None:
        try:
            super().build_extension(ext)
        except Exception as exc:  # pragma: no cover - toolchain-dependent
            self._warn(exc)

    @staticmethod
    def _warn(exc: Exception) -> None:
        import sys

        print(
            "warning: veeksha_native (optional C++ engine) failed to build "
            f"({exc}); the Python transport will be used instead.",
            file=sys.stderr,
        )


setup(ext_modules=ext_modules, cmdclass={"build_ext": OptionalBuildExt})
