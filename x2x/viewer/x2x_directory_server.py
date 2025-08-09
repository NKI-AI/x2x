#!/usr/bin/env python
#
# x2x_directory_server - Web application for viewing multiple slides with AI explanations
#
# Copyright (c) 2010-2015 Carnegie Mellon University
# Copyright (c) 2021-2024 Benjamin Gilbert
# Copyright (c) 2024-2025 Yoni Schirris (modifications for x2x project)
#
# Modified 2024-2025 for x2x project - added functionality for thumbnail caching,
# patch processing, and AI-based explanations integration
#
# This library is free software; you can redistribute it and/or modify it
# under the terms of version 3 or later of the GNU Lesser General Public License
# as published by the Free Software Foundation.
#
# This library is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
# License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

from __future__ import annotations

import hashlib
import logging
import os
from io import BytesIO
from pathlib import Path, PurePath
from typing import Any, Dict

import dlup
import pandas as pd
from flask import (
    Response,
    abort,
    make_response,
    render_template,
    send_from_directory,
    url_for,
)
from tqdm import tqdm

from x2x.utils.logger import get_logger

if os.name == "nt":
    _dll_path = os.getenv("OPENSLIDE_PATH")
    if _dll_path is not None:
        with os.add_dll_directory(_dll_path):
            import openslide
    else:
        import openslide
else:
    import openslide

from openslide import OpenSlide, OpenSlideError
from openslide.deepzoom import DeepZoomGenerator

from x2x.viewer.x2x_base_server import BaseX2XServer

# Initialize with default level, will be updated from config in create_app
log = get_logger(__name__, level=logging.INFO)


class X2XDirectoryServer(BaseX2XServer):
    def __init__(self, import_name: str) -> None:
        super().__init__(import_name)
        self.basedir: Path = None
        self.thumbnail_cache = {}  # In-memory cache
        self.thumbnail_cache_dir = None  # Will be set in create_app

    def get_slide_path(self, data: Dict[str, Any]) -> Path:
        """Get slide path for multi-slide server"""
        slide_path = self.basedir / data.get("slide_path", "")
        if not slide_path.is_file():
            raise ValueError(f"Slide file not found: {slide_path}")
        if slide_path.parts[: len(self.basedir.parts)] != self.basedir.parts:
            raise ValueError("Invalid slide path")
        return slide_path

    def get_thumbnail(self, slide_path: Path) -> str | None:
        """Get cached thumbnail or generate a new one"""
        cache_key = str(slide_path)

        # Check in-memory cache first
        if cache_key in self.thumbnail_cache:
            return self.thumbnail_cache[cache_key]

        # Create cache filename using SHA-256 hash
        cache_filename = hashlib.sha256(cache_key.encode()).hexdigest() + ".jpg"
        cache_path = self.thumbnail_cache_dir / cache_filename

        # Check disk cache
        if cache_path.exists():
            try:
                return url_for("serve_thumbnail", filename=cache_filename)
            except Exception as e:
                log.error(f"Error reading cached thumbnail for {slide_path}: {e}")

        # Generate new thumbnail
        try:
            slide_image = dlup.SlideImage.from_file_path(slide_path)
            _scaling = slide_image.get_scaling(mpp=32)
            target_size = slide_image.get_scaled_size(_scaling)
            thumbnail = slide_image.get_thumbnail(target_size)
            thumbnail.save(cache_path, format="JPEG", quality=70)
            thumbnail_url = url_for("serve_thumbnail", filename=cache_filename)
            self.thumbnail_cache[cache_key] = thumbnail_url
            return thumbnail_url
        except Exception as e:
            log.error(f"Error generating thumbnail for {slide_path}: {e}")
            return None


def create_app(
    config: Dict[str, Any] | None = None,
    config_file: Path | None = None,
) -> X2XDirectoryServer:
    # Create and configure app
    app = X2XDirectoryServer(__name__)

    # Add multi-slide specific config
    app.config.update(
        SLIDE_DIR=".",  # Only set slide directory, LOG_LEVEL comes from DEFAULT_CONFIG
    )

    app.config.from_envvar("DEEPZOOM_MULTISERVER_SETTINGS", silent=True)
    if config_file is not None:
        app.config.from_pyfile(config_file)
    if config is not None:
        app.config.from_mapping(config)

    # Update log level from config
    global log
    log_level = app.config[
        "LOG_LEVEL"
    ]  # This will use the value from DEFAULT_CONFIG if not overridden
    log = get_logger(__name__, level=getattr(logging, log_level))

    # Also set the root logger's level to ensure proper propagation
    logging.getLogger().setLevel(getattr(logging, log_level))

    # Set up base directory
    app.basedir = Path(app.config["SLIDE_DIR"]).resolve(strict=True)

    # Set up thumbnail cache directory
    app.thumbnail_cache_dir = (
        Path(app.config["LOG_DIR"]) / app.config["EXPERIMENT_NAME"] / "thumbnails"
    )
    app.thumbnail_cache_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Using thumbnail cache directory: {app.thumbnail_cache_dir}")

    # Add route to serve thumbnails
    @app.route("/thumbnails/<path:filename>")
    def serve_thumbnail(filename):
        return send_from_directory(app.thumbnail_cache_dir, filename)

    # Initialize patch processor
    app.init_patch_processor()

    # Set up routes
    @app.route("/")
    def index() -> str:
        root_dir = _Directory(app.basedir, app)
        return render_template("files.html", root_dir=root_dir.to_dict())

    @app.route("/<path:path>")
    def slide(path: str) -> str:
        try:
            slide_path = (app.basedir / PurePath(path)).resolve(strict=True)
            if slide_path.parts[: len(app.basedir.parts)] != app.basedir.parts:
                abort(404)

            osr = OpenSlide(slide_path)
            slide = DeepZoomGenerator(
                osr,
                tile_size=app.config["DEEPZOOM_TILE_SIZE"],
                overlap=app.config["DEEPZOOM_OVERLAP"],
                limit_bounds=app.config["DEEPZOOM_LIMIT_BOUNDS"],
            )

            try:
                mpp_x = osr.properties[openslide.PROPERTY_NAME_MPP_X]
                mpp_y = osr.properties[openslide.PROPERTY_NAME_MPP_Y]
                mpp = (float(mpp_x) + float(mpp_y)) / 2
            except (KeyError, ValueError):
                mpp = 0

            # Get slide info when slide is first loaded

            slide_url = url_for("dzi", path=path)
            return render_template(
                "slide-multipane.html",
                config={
                    "TARGET_MPP": app.config["TARGET_MPP"],
                    "TILE_SIZE": app.config["TILE_SIZE"],
                },
                slide_url=slide_url,
                associated={},
                properties=osr.properties,
                slide_mpp=mpp,
            )
        except (OpenSlideError, OSError):
            abort(404)

    @app.route("/<path:path>.dzi")
    def dzi(path: str) -> Response:
        try:
            slide_path = (app.basedir / PurePath(path)).resolve(strict=True)
            if slide_path.parts[: len(app.basedir.parts)] != app.basedir.parts:
                abort(404)

            osr = OpenSlide(slide_path)
            slide = DeepZoomGenerator(
                osr,
                tile_size=app.config["DEEPZOOM_TILE_SIZE"],
                overlap=app.config["DEEPZOOM_OVERLAP"],
                limit_bounds=app.config["DEEPZOOM_LIMIT_BOUNDS"],
            )

            format = app.config["DEEPZOOM_FORMAT"]
            resp = make_response(slide.get_dzi(format))
            resp.mimetype = "application/xml"
            return resp
        except (OpenSlideError, OSError):
            abort(404)

    @app.route("/<path:path>_files/<int:level>/<int:col>_<int:row>.<format>")
    def tile(path: str, level: int, col: int, row: int, format: str) -> Response:
        try:
            slide_path = (app.basedir / PurePath(path)).resolve(strict=True)
            if slide_path.parts[: len(app.basedir.parts)] != app.basedir.parts:
                abort(404)

            osr = OpenSlide(slide_path)
            slide = DeepZoomGenerator(
                osr,
                tile_size=app.config["DEEPZOOM_TILE_SIZE"],
                overlap=app.config["DEEPZOOM_OVERLAP"],
                limit_bounds=app.config["DEEPZOOM_LIMIT_BOUNDS"],
            )

            format = format.lower()
            if format != "jpeg" and format != "png":
                abort(404)
            try:
                tile = slide.get_tile(level, (col, row))
            except ValueError:
                abort(404)

            buf = BytesIO()
            tile.save(
                buf,
                format,
                quality=app.config["DEEPZOOM_TILE_QUALITY"],
            )
            resp = make_response(buf.getvalue())
            resp.mimetype = "image/%s" % format
            return resp
        except (OpenSlideError, OSError):
            abort(404)

    # Set up the extract_patch route
    app.setup_extract_patch_route()
    app.setup_processed_patches_route()

    return app


class _SlideFile:
    def __init__(self, relpath: PurePath, app: X2XDirectoryServer) -> None:
        self.name = relpath.name
        self.url_path = relpath.as_posix()
        self.clinical_data = None
        self.predictions = None  # Initialize predictions
        self.targets = None  # Initialize targets
        self.thumbnail = None  # Initialize thumbnail

        try:
            # Generate thumbnail
            slide_path = (app.basedir / relpath).resolve(strict=True)
            if slide_path.is_file():
                self.thumbnail = app.get_thumbnail(slide_path)

            # Get relative path starting with ./
            relative_path = "./" + self.url_path

        except Exception as e:
            log.error(f"Error processing slide {self.name}: {e}")

    def to_dict(self):
        return {
            "name": self.name,
            "url_path": self.url_path,
            "predictions": self.predictions,
            "targets": self.targets,
            "clinical_data": self.clinical_data,
            "thumbnail": self.thumbnail,
        }


class _Directory:
    _DEFAULT_RELPATH = PurePath(".")
    SLIDE_EXTENSIONS = {
        ".svs",
        ".tif",
        ".tiff",
        ".ndpi",
        ".vms",
        ".vmu",
        ".scn",
        ".mrxs",
        ".bif",
    }

    def __init__(
        self,
        basedir: Path,
        app: X2XDirectoryServer,
        relpath: PurePath = _DEFAULT_RELPATH,
    ) -> None:
        self.name = relpath.name
        self.slides = []  # Store all slides

        # First collect all slide paths
        slide_paths = []

        def collect_slide_paths(path: PurePath):
            for cur_path in sorted((basedir / path).iterdir()):
                cur_relpath = path / cur_path.name
                if cur_path.is_dir():
                    collect_slide_paths(cur_relpath)
                elif (
                    cur_path.suffix.lower() in self.SLIDE_EXTENSIONS
                    and OpenSlide.detect_format(cur_path)
                ):
                    slide_paths.append(cur_relpath)

        collect_slide_paths(relpath)

        # Then process slides with progress bar
        if slide_paths:
            log.info(f"Generating thumbnails for {len(slide_paths)} slides...")
            for cur_relpath in tqdm(slide_paths, desc="Generating thumbnails"):
                self.slides.append(_SlideFile(cur_relpath, app))

        # Create one big table from all slides
        if self.slides:
            rows = []
            for slide in self.slides:
                relative_path = "./" + slide.url_path
                row = {
                    "Thumbnail": slide.thumbnail or "",  # Add thumbnail column
                    "Slide": f'<a href="{slide.url_path}">{slide.name}</a>',
                }

                rows.append(row)

            # Convert to DataFrame and then to HTML
            df = pd.DataFrame(rows)

            self.table = df.to_html(
                index=False,
                classes="info-table display",
                float_format=lambda x: "%.2f" % x if pd.notnull(x) else "",
                escape=False,
                table_id="slides-table",
            )
        else:
            self.table = "<p>No slides found</p>"

    def to_dict(self):
        return {"name": self.name, "table": self.table}
