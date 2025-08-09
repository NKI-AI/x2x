import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Union

import dlup
from flask import Flask, Response, abort, jsonify, request, send_from_directory
from PIL import Image

from x2x.utils.logger import get_logger
from x2x.viewer.patch_processor import PatchProcessor


class BaseX2XServer(Flask):
    def __init__(self, import_name: str) -> None:
        super().__init__(import_name)

        DEEPZOOM_CONFIG = {
            "DEEPZOOM_FORMAT": "jpeg",
            "DEEPZOOM_TILE_SIZE": 254,
            "DEEPZOOM_OVERLAP": 1,
            "DEEPZOOM_LIMIT_BOUNDS": True,
            "DEEPZOOM_TILE_QUALITY": 75,
        }

        X2X_DEFAULT_CONFIG = {
            "DEVICE": "cpu",
            "EXPERIMENT_NAME": "DEBUG",
            "TARGET_MPP": 1.14,
            "TILE_SIZE": 224,
            "LOG_LEVEL": "DEBUG",
        }

        # Default configuration
        self.config.from_mapping(
            **DEEPZOOM_CONFIG,
            **X2X_DEFAULT_CONFIG,
        )

        global logger
        logger = get_logger(__name__, level=getattr(logging, self.config["LOG_LEVEL"]))

        self.setup_routes()

    def init_patch_processor(self) -> None:
        """Initialize the patch processor with current configuration"""

        self.patch_processor = PatchProcessor(
            cfg=self.config,
        )

    def setup_extract_patch_route(self) -> None:
        """Set up the extract_patch route that both servers share"""

        @self.route("/extract_patch", methods=["POST"])
        def extract_patch() -> Union[Dict[str, Any], tuple[Dict[str, Any], int]]:
            """Extract and process a patch from a slide image.

            Expected JSON input:
            {
                "x": int,      # X coordinate at target MPP (e.g. 0.5mpp)
                "y": int,      # Y coordinate at target MPP (e.g. 0.5mpp)
                "mpp": float,  # Microns per pixel (e.g. 0.5)
                "slide_path": str  # Path/identifier of the slide (implementation specific)
            }

            Returns
            -------
                Union[dict[str, Any], tuple[dict[str, Any], int]]: Either:
                - Success response: {
                    "status": "success",
                    "scores": Union[list[float], list[list[float]]],  # 1D or 2D array of model scores
                    "probabilities": Union[list[float], list[list[float]]],  # 1D or 2D array of probabilities
                    "attention": Optional[list[float]],  # Attention scores if available
                    "save_path": str,  # Path where processed patch was saved
                    "images": dict[str, str]  # Dict of image identifiers to base64 encoded images
                }
                - Error response: ({"status": "error", "message": str}, int)
                  where int is the HTTP status code (400, 404, or 500)
            """
            data = request.get_json()
            if not data:
                return {"status": "error", "message": "No data received"}, 400

            logger.debug("Received patch request:", data)

            try:
                # Get coordinates and mpp from request
                try:

                    x = int(data["x"])  # @ 0.5mpp
                    y = int(data["y"])  # @ 0.5mpp
                    mpp = float(data["mpp"])  # =0.5mpp
                except (KeyError, ValueError) as e:
                    return {
                        "status": "error",
                        "message": f"Invalid coordinates or mpp: {str(e)}",
                    }, 400

                # Get slide path - this will be different for single/multi server
                try:
                    slide_path = self.get_slide_path(data)
                except Exception as e:
                    return {"status": "error", "message": str(e)}, 404

                # Open slide and extract patch
                try:
                    slide = dlup.SlideImage.from_file_path(slide_path)
                    # Get the actual MPP from the slide
                    actual_mpp = (
                        slide.mpp if isinstance(slide.mpp, float) else slide.mpp[0]
                    )
                    logger.debug(
                        f"Requested MPP: {mpp}, Actual slide MPP: {actual_mpp}"
                    )

                    patch_size = int(self.config["TILE_SIZE"])
                    patch = slide.read_region(
                        (x, y),
                        scaling=slide.get_scaling(mpp=float(self.config["TARGET_MPP"])),
                        size=(patch_size, patch_size),
                    )
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Error reading slide or extracting patch: {str(e)}",
                    }, 500

                # Process the patch through our model
                try:
                    results = self.patch_processor.process_patch(
                        patch=patch,
                        slide_name=Path(slide_path).stem,
                        location=(x, y),
                        mpp=mpp,
                        save_format="jpeg",
                    )
                    logger.debug(
                        f"Patch processing results: {list(results.keys())}"
                    )  # Debug log
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Error processing patch: {str(e)}",
                    }, 500

                response = {
                    "status": "success",
                    "scores": results["scores"],
                    "probabilities": results["probabilities"],
                    "attention": results["attention"],
                    "save_path": results["save_path"],
                    "images": results["images"],
                }
                logger.debug("Sending response with keys: %s", response.keys())
                return response

            except Exception as e:
                logger.error(f"Unexpected error processing patch: {e}")
                import traceback

                traceback.print_exc()
                logger.error(traceback.format_exc())
                return {
                    "status": "error",
                    "message": f"Unexpected error: {str(e)}",
                }, 500

    def setup_processed_patches_route(self) -> None:
        """Set up route to serve processed patch images"""

        @self.route(
            f"{self.config['LOG_DIR']}/{self.config['EXPERIMENT_NAME']}/<path:filename>"
        )
        def serve_processed_patch(filename: str) -> Response:
            # Construct the full path using LOG_DIR and EXPERIMENT_NAME
            log_dir = Path(self.config["LOG_DIR"]) / self.config["EXPERIMENT_NAME"]

            # Ensure absolute path
            if not log_dir.is_absolute():
                log_dir = Path.cwd() / log_dir

            # Force logging to print
            logger.debug(f"Current working directory: {Path.cwd()}")
            logger.debug(f"Log dir: {log_dir}")
            logger.debug(f"Requested filename: {filename}")
            logger.debug(f"Full path: {log_dir}/{filename}")
            logger.debug(f"File exists: {(log_dir/filename).exists()}")

            if not (log_dir / filename).exists():
                logger.error(f"File not found: {log_dir}/{filename}")
                abort(404)

            return send_from_directory(log_dir, filename)

    @abstractmethod
    def get_slide_path(self, data: dict[str, Any]) -> Path:
        """
        Get the slide path from the request data.
        To be implemented by subclasses.

        Parameters
        ----------
        data : dict[str, Any]
            Request data containing slide path information

        Returns
        -------
        Path
            Path to the slide file

        Raises
        ------
        NotImplementedError
            If not implemented by subclass
        """
        raise NotImplementedError

    def setup_routes(self):
        """Set up basic routes that both servers need"""

        @self.route("/analyze_single_image", methods=["POST"])
        def analyze_single_image():
            try:
                # Initialize patch processor if not done yet
                if not hasattr(self, "patch_processor"):
                    self.init_patch_processor()

                data = request.get_json()
                if not data or "image_path" not in data:
                    return jsonify({"error": "image_path is required"}), 400

                image_path = Path(data["image_path"])
                if not image_path.exists():
                    return jsonify({"error": f"Image not found: {image_path}"}), 404

                # Load and process the image
                try:
                    image = Image.open(image_path)
                except Exception as e:
                    return jsonify({"error": f"Failed to load image: {str(e)}"}), 400

                # Process the patch
                results = self.patch_processor.process_patch(
                    patch=image,
                    slide_name=image_path.stem,
                    location=(0, 0),
                    mpp=0.5,
                    save_format="jpeg",
                )

                return jsonify({"status": "success", "results": results})

            except Exception as e:
                logger.error(f"Error processing single image: {e}")
                import traceback

                logger.error(traceback.format_exc())
                return jsonify({"error": str(e)}), 500
