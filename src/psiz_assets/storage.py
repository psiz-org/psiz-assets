# -*- coding: utf-8 -*-
# Copyright 2026 The PsiZ Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Storage backend interfaces for resolving dataset and model asset URIs."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, urlparse

from .core import AssetError


class AssetStore(Protocol):
    """Backend capable of resolving a published asset URI locally."""

    def resolve(self, uri: str) -> Path:
        ...


class LocalAssetStore:
    """Resolve local paths and ``file://`` URIs."""

    def resolve(self, uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme not in {"", "file"}:
            raise AssetError(f"local storage cannot resolve URI scheme: {parsed.scheme}")
        path = Path(parsed.path if parsed.scheme == "file" else uri).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"asset does not exist: {path}")
        return path


class HuggingFaceAssetStore:
    """Resolve ``hf://org/repo[/subpath]`` URIs using optional huggingface_hub."""

    def __init__(self, *, revision: str = "main", cache_dir: str | Path | None = None):
        self.revision = revision
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None

    def resolve(self, uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != "hf" or not parsed.netloc:
            raise AssetError(f"invalid Hugging Face asset URI: {uri}")
        path_parts = [part for part in parsed.path.split("/") if part]
        if not path_parts:
            raise AssetError(f"invalid Hugging Face asset URI: {uri}")
        revision = parse_qs(parsed.query).get("revision", [self.revision])[0]
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise ImportError(
                "huggingface_hub is required to resolve hf:// asset URIs"
            ) from error
        repo_id = f"{parsed.netloc}/{path_parts[0]}"
        subpath = "/".join(path_parts[1:])
        repo_type = "model" if not subpath else "dataset"
        download_kwargs = {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "revision": revision,
            "cache_dir": self.cache_dir,
        }
        if subpath:
            download_kwargs["allow_patterns"] = [subpath, f"{subpath}/**"]
        root = Path(snapshot_download(**download_kwargs))
        if not subpath:
            return root
        resolved = root / subpath
        if not resolved.exists():
            raise FileNotFoundError(f"asset path is missing from Hugging Face repo: {uri}")
        return resolved