import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from x2x.oai.oai_utils import (
    create_prediction_prompt,
    load_api_key,
    load_prompt,
    save_results,
)
from x2x.utils.logger import get_logger

logger = get_logger()


def evaluate_explanation(
    explanation_path: str,
    image_dir: str,
    log_dir: str,
    image_prompt_path: str,
):
    """Evaluate the explanation for a given image.

    Parameters
    ----------
        explanation_path (str): Path to the explanation file
        image_dir (str): Path to the directory containing the images
        log_dir (str): Path to the directory to save the results
        image_prompt_path (str): Path to the image prompt file

    Returns
    -------
        None

    Raises:
    -------
        ValueError: If no conclusion is found in the prediction of score
    """

    os.environ["OPENAI_API_KEY"] = load_api_key()
    client = OpenAI()

    explanation = load_prompt(explanation_path)
    logger.info(f"Loaded explanation: {explanation}")

    image_paths = Path(image_dir).glob("**/macenko.jpeg")

    results = []
    for image_path in image_paths:

        try:
            metadata = json.load(open(image_path.parent / "metadata.json"))
            logit = metadata["raw_logits"]["label_0"]
            attention = (
                metadata["attention_logits"]["label_0"][0]
                if len(metadata["attention_logits"]["label_0"]) == 1
                else metadata["attention_logits"]["label_0"]
            )
            logger.info(f"Image: {image_path}")
            logger.info(f"Metadata: {metadata}")

            # Combines image_path, explanation, and prediction of score prompt into a single prompt
            system_prompt_message_for_prediction_of_score, image_description = (
                create_prediction_prompt(
                    image_path, client, explanation_path, log_dir, image_prompt_path
                )
            )

            prediction_of_score = (
                client.chat.completions.create(
                    model="gpt-4o",  # Hardcoded for current experiments
                    messages=system_prompt_message_for_prediction_of_score,
                    stream=False,
                    max_completion_tokens=16384,
                )
                .choices[0]
                .message.content
            )
            match = re.search(r"CONCLUSION=(.+)$", prediction_of_score, re.MULTILINE)

            if not match:
                raise ValueError("No conclusion found in the prediction of score")

            conclusion = match.group(1)
            binary_logit = 1 if logit > -0.135 else 0

            results.append(
                {
                    "image_path": str(image_path),
                    "conclusion": conclusion,
                    "logit": logit,
                    "attention": attention,
                    "binary_logit": binary_logit,
                    "image_description": image_description,
                    "prediction_of_score": prediction_of_score,
                }
            )
            logger.info(f"Prediction of score: {prediction_of_score}")
        except Exception as e:
            logger.error(f"Error: {e}")
            prediction_of_score = ""

    # Generate timestamp and save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(
        results=results,
        log_dir=log_dir,
        explanation=explanation,
        timestamp=timestamp,
        image_prompt_path=image_prompt_path,
        system_prompt_message_for_prediction_of_score=system_prompt_message_for_prediction_of_score[
            0
        ],  # The first one is the system prompt
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate explanation")
    parser.add_argument(
        "--repo_root",
        type=str,
        required=False,
        help="Path to the repo root",
        default=".",  # current directory, expected to be run from the root, i.e., `python tools/evalaute_explanation.py`
    )
    parser.add_argument(
        "--explanation",
        type=str,
        required=False,
        help="Absolute (or relative to repo_root) path to the explanation file",
        default="prompts/explanation/detailed_analysis.txt",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=False,
        default="logs/evaluate_explanation",
        help="Absolute (or relative to repo_root) path to the log dir",
    )
    parser.add_argument(
        "--image_description_prompt",
        type=str,
        required=False,
        default="prompts/image_description/system_prompt_image_4o.txt",
        help="Absolute (or relative to repo_root) path to the image prompt file",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Absolute (or relative to repo_root) path to the dir containing images",
    )

    args = parser.parse_args()

    def absolute_or_relative_path(path: str, repo_root: str):
        return Path(path) if Path(path).is_absolute() else Path(repo_root) / path

    evaluate_explanation(
        explanation_path=absolute_or_relative_path(args.explanation, args.repo_root),
        image_dir=absolute_or_relative_path(args.image_dir, args.repo_root),
        log_dir=absolute_or_relative_path(args.log_dir, args.repo_root),
        image_prompt_path=absolute_or_relative_path(
            args.image_description_prompt, args.repo_root
        ),
    )
