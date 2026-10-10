"""Dependency-isolated helpers; these tests do not validate hardware kernels."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]


def load_platform_modules(monkeypatch, torch):
    """Execute real platform modules in a private package with a stub torch."""
    package = ModuleType("_flagos_review_platform")
    package.__path__ = [str(ROOT / "verl/plugin/platform")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, "torch", torch)
    for name in ("platform_base", "platform_flagos", "platform_manager"):
        full_name = f"{package.__name__}.{name}"
        spec = importlib.util.spec_from_file_location(full_name, Path(package.__path__[0]) / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full_name, module)
        spec.loader.exec_module(module)
        setattr(package, name, module)
    return package.platform_flagos, package.platform_manager


def source_functions(path, names, namespace, assignments=()):
    """Execute exact function ASTs without importing unavailable ML dependencies.

    Methods are lifted unchanged out of their classes. This validates Python
    control flow only, not integration with real Ray/Torch/Transformers modules.
    """
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in assignments for target in node.targets
        ):
            selected.append(node)
        elif isinstance(node, ast.ClassDef):
            selected.extend(child for child in node.body if isinstance(child, ast.FunctionDef) and child.name in names)
    assert {node.name for node in selected if isinstance(node, ast.FunctionDef)} == set(names)
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(ROOT / path), "exec"), namespace)
    return namespace
