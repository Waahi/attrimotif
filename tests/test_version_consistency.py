# -*- coding: utf-8 -*-
"""The package version is asserted in five places plus the manuscript's Code
metadata table; a release where they disagree is a release that misreports what
was archived. This pins them together."""
from __future__ import annotations

import json
import os
import re

import attrimotif as am

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name):
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        # This fired once for real: .zenodo.json was missing from MANIFEST.in,
        # so the sdist shipped this test together with a file it did not
        # contain, and anyone running the downloaded tests saw a bare
        # FileNotFoundError. Say what is actually wrong instead.
        raise AssertionError(
            f"{name} is not present next to the package. If this is a source "
            f"distribution, {name} is missing from MANIFEST.in: the test suite "
            f"is shipped but the file it checks is not.")
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_all_declared_versions_agree():
    v = am.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", v), v
    assert re.search(r'^version = "%s"$' % re.escape(v), _read("pyproject.toml"), re.M)
    assert re.search(r"^version: %s$" % re.escape(v), _read("CITATION.cff"), re.M)
    assert json.loads(_read("codemeta.json"))["version"] == v
    assert json.loads(_read(".zenodo.json"))["version"] == v


def test_changelog_documents_the_current_version():
    assert f"## [{am.__version__}]" in _read("CHANGELOG.md")
