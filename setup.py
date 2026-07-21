"""Build the optional native (C++) benchmark native loop on install.

``veeksha_native`` is a pybind11 free-threaded extension
(``py::mod_gil_not_used``) that owns the microsecond-sensitive benchmark main
loop: native scheduler, dispatch, reactor transports, completion ack. It is
OPTIONAL: if a C++ toolchain or pybind11 is unavailable, the build degrades
gracefully and ``veeksha.native.is_available()`` returns False so callers fall
back to the Python main loop. Metadata otherwise comes from pyproject.toml.
"""

from __future__ import annotations

from setuptools import setup
from setuptools.command.build_ext import build_ext

try:
    from pybind11.setup_helpers import Pybind11Extension

    ext_modules = [
        Pybind11Extension(
            "veeksha.native.veeksha_native",
            [
                "veeksha/native/src/native_loop.cpp",
                # vendored llhttp (HTTP/1.1 response framing; MIT) — plain C
                # sources compiled into the same extension
                "veeksha/native/third_party/llhttp/api.c",
                "veeksha/native/third_party/llhttp/http.c",
                "veeksha/native/third_party/llhttp/llhttp.c",
            ],
            include_dirs=["veeksha/native/third_party/llhttp"],
            cxx_std=17,
            extra_compile_args=["-pthread"],
            extra_link_args=["-pthread"],
        )
    ]
except Exception:  # pragma: no cover - pybind11 missing at build time
    ext_modules = []


class OptionalBuildExt(build_ext):
    """Never fail the wheel because the native extension didn't compile."""

    def build_extensions(self) -> None:
        # Pybind11Extension puts C++-only flags (-std=c++17) in
        # extra_compile_args, which distutils applies to every source. Strip
        # -std=* when compiling the vendored llhttp *.c files so clang/gcc
        # don't reject C++ standards flags in C mode.
        compiler = self.compiler
        if hasattr(compiler, "_compile"):  # unix-style compilers

            original = compiler._compile

            def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
                if src.endswith(".c"):
                    extra_postargs = [
                        a for a in extra_postargs if not a.startswith("-std=")
                    ]
                return original(obj, src, ext, cc_args, extra_postargs, pp_opts)

            compiler._compile = _compile
        super().build_extensions()

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
            "warning: veeksha_native (optional C++ main loop) failed to build "
            f"({exc}); the Python main loop will be used instead.",
            file=sys.stderr,
        )


setup(ext_modules=ext_modules, cmdclass={"build_ext": OptionalBuildExt})
