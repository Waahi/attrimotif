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
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
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
