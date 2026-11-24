from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import re
import sys
import typing
from pathlib import Path
from typing import Any

PATCHED_MODULES = frozenset(
    {
        "lerobot.utils.io_utils",
        "lerobot.processor.pipeline",
        "lerobot.datasets.streaming_dataset",
        "lerobot.motors.motors_bus",
    }
)

_TYPE_ALIAS = re.compile(r"^(\s*)type\s+([A-Za-z_]\w*)\s*=\s*", re.MULTILINE)
_DEF_PARAMS = re.compile(r"^(\s*)(def|class)\s+([A-Za-z_]\w*)\s*\[", re.MULTILINE)

_CLASS_GETITEM = (
    "\n    def __class_getitem__(cls, item):  # PEP 695 erasure shim\n"
    "        return cls\n"
)


def _strip_bracket(source: str, open_index: int) -> tuple[str, str]:
    depth = 0
    for index in range(open_index, len(source)):
        character = source[index]
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                return source[:open_index] + source[index + 1 :], source[open_index + 1 : index]
    raise SyntaxError("unbalanced type-parameter bracket in PEP 695 shim")


def _parameter_names(inside: str) -> list[str]:
    names = []
    depth = 0
    current = ""
    for character in inside + ",":
        if character in "[({":
            depth += 1
        elif character in "])}":
            depth -= 1
        if character == "," and depth == 0:
            token = current.split(":")[0].strip().lstrip("*")
            if token:
                names.append(token)
            current = ""
        else:
            current += character
    return names


def rewrite_source(source: str) -> str:
    source = _TYPE_ALIAS.sub(r"\1\2 = ", source)
    generic_classes: list[str] = []
    parameters: list[str] = []
    while True:
        match = _DEF_PARAMS.search(source)
        if match is None:
            break
        source, inside = _strip_bracket(source, match.end() - 1)
        for name in _parameter_names(inside):
            if name not in parameters:
                parameters.append(name)
        if match.group(2) == "class":
            generic_classes.append(match.group(3))
    for name in generic_classes:
        pattern = re.compile(
            rf"^(\s*)class\s+{re.escape(name)}\s*(\([^)]*\))?\s*:[ \t]*\n", re.MULTILINE
        )

        def _substitute(match: re.Match[str]) -> str:
            bases = match.group(2) or ""
            body = _CLASS_GETITEM.replace("\n    ", "\n" + match.group(1) + "    ")
            return f"{match.group(1)}class {name}{bases}:{body}"

        source = pattern.sub(_substitute, source, count=1)
    if parameters:
        preamble = "from typing import TypeVar as _ShimTypeVar\n" + "".join(
            f"{name} = _ShimTypeVar({name!r})\n" for name in parameters
        )
        source = _insert_after_future_imports(source, preamble)
    return source


def _insert_after_future_imports(source: str, preamble: str) -> str:
    lines = source.splitlines(keepends=True)
    position = 0
    for index, line in enumerate(lines):
        if line.startswith("from __future__"):
            position = index + 1
    return "".join(lines[:position]) + preamble + "".join(lines[position:])


class _RewritingLoader(importlib.machinery.SourceFileLoader):
    def source_to_code(self, data: Any, path: Any, *, _optimize: int = -1) -> Any:
        text = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data
        return compile(rewrite_source(text), path, "exec", dont_inherit=True, optimize=_optimize)

    def get_code(self, fullname: str) -> Any:
        return self.source_to_code(self.get_data(self.get_filename(fullname)),
                                   self.get_filename(fullname))


class _Finder(importlib.abc.MetaPathFinder):

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname not in PATCHED_MODULES:
            return None
        parts = fullname.split(".")
        roots = list(path) if path is not None else sys.path
        tails = [Path(parts[-1] + ".py")] if path is not None else [
            Path(*parts).with_suffix(".py")
        ]
        for root in roots:
            for tail in tails:
                candidate = Path(str(root)) / tail
                if candidate.exists():
                    loader = _RewritingLoader(fullname, str(candidate))
                    return importlib.util.spec_from_file_location(
                        fullname, str(candidate), loader=loader
                    )
        return None


_INSTALLED = False


def install() -> dict[str, Any]:
    global _INSTALLED
    shimmed = []
    import typing_extensions

    for name in ("Self", "TypeAlias", "Unpack", "override", "ParamSpec", "TypeVarTuple"):
        if not hasattr(typing, name) and hasattr(typing_extensions, name):
            setattr(typing, name, getattr(typing_extensions, name))
            shimmed.append(name)
    if not _INSTALLED:
        sys.meta_path.insert(0, _Finder())
        _INSTALLED = True
    return {
        "python_version": sys.version.split()[0],
        "typing_names_backfilled": shimmed,
        "pep695_modules_rewritten": sorted(PATCHED_MODULES),
        "note": "lerobot source is read-only; rewriting happens in memory at import time",
    }
