"""Selectors: one way to name modules of a model, used by every edit and setting in the library.

A selector is one of

    "model.layers.3.mlp"        a module name
    "*.mlp", "*.layers.1?.*"    a glob over full module names (fnmatch, case-sensitive)
    "re:layers\\.[0-3]\\."      a regular expression searched in module names (anchor with ^ and $ for a full match)
    torch.nn.LayerNorm          a module class (instances of it and of subclasses)
    module                      a module object of the model
    [s1, s2, ...]               any of several selectors
    fn(name, module) -> bool    a predicate

The model itself has the empty name: "*" (and a class or predicate that accepts it) selects it, i.e. everything;
other patterns select submodules.
"""
from __future__ import annotations

import fnmatch
import re
from collections.abc import Callable, Iterable
from typing import Union

import torch

__all__ = ["Selector", "matches", "select"]

Selector = Union[str, type, torch.nn.Module, Iterable, Callable[[str, torch.nn.Module], bool]]


def matches(selector: Selector, name: str, module: torch.nn.Module) -> bool:
    """Whether ``selector`` selects the module ``module`` named ``name``."""
    if isinstance(selector, str):
        if selector.startswith("re:"):
            return re.search(selector[3:], name) is not None
        return fnmatch.fnmatchcase(name, selector)
    if isinstance(selector, type):
        return isinstance(module, selector)
    if isinstance(selector, torch.nn.Module):
        return module is selector
    if callable(selector):
        return bool(selector(name, module))
    return any(matches(s, name, module) for s in selector)


def select(model: torch.nn.Module, selector: Selector, outermost: bool = False) -> list[tuple[str, torch.nn.Module]]:
    """(name, module) of every module of ``model`` that ``selector`` selects, in model order. ``outermost``: drop
    modules nested inside another selected module."""
    out = [(n, m) for n, m in model.named_modules() if matches(selector, n, m)]
    if outermost:
        names = {n for n, _ in out}
        out = [(n, m) for n, m in out if not any(n.startswith(o + ".") or (o == "" and n) for o in names if o != n)]
    return out
