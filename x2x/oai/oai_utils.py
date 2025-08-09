import base64
import hashlib
import json
import logging
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI

from x2x.utils.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def load_api_key() -> str:
    """Load OpenAI API key from ~/.openai.

    Returns:
        str: OpenAI API key

    Raises:
        Exception: If the OpenAI API key is not found
    """
    key_path = os.path.expanduser("~/.openai")
    try:
        with open(key_path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise Exception(
            "OpenAI API key not found. Please create ~/.openai file with your API key."
        )


def load_prompt(filename: str) -> str:
    """Load prompt from file.

    Args:
        filename: Path to the prompt file

    Returns:
        str: Prompt
    """
    try:
        with open(filename, "r") as f:
            logger.info(f"Loading prompt from: {filename}")
            return f.read().strip()
    except FileNotFoundError:
        raise Exception(f"Prompt file {filename} not found.")


def get_mime_type(file_path: str) -> str:
    """Get MIME type based on file extension.

    Args:
        file_path: Path to the file

    Returns:
        str: MIME type
    """
    ext = os.path.splitext(file_path)[1].lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mime_types.get(ext, "image/jpeg")


def create_request_hash(
    messages: list[dict], model: str = "gpt-4o", max_tokens: int = 500
) -> str:
    """
    Create a unique hash for the GPT request based on its parameters.

    Args:
        messages: List of message dictionaries for the GPT request
        model: Model name (default: gpt-4o)
        max_tokens: Maximum tokens for completion (default: 500)

    Returns:
        str: Hex digest of the hash
    """
    # Convert messages to a stable string representation
    request_str = json.dumps(
        {"messages": messages, "model": model, "max_tokens": max_tokens}, sort_keys=True
    )

    return hashlib.sha256(request_str.encode()).hexdigest()


def get_cache_path(log_dir: Path, request_hash: str) -> Path:
    """
    Get the cache file path for a given request hash.

    Args:
        log_dir: Path to the log directory
        request_hash: Hash of the request parameters

    Returns:
        Path: Path to the cache file
    """
    cache_dir = Path(log_dir) / "cache"
    return cache_dir / f"{request_hash}.json"


def save_to_cache(cache_path: Path, response: str) -> None:
    """
    Save response to cache file.

    Args:
        cache_path: Path to save the cache file
        response: Response content to cache

    Returns:
        None
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"response": response}, f)


def load_from_cache(cache_path: Path) -> str | None:
    """
    Load response from cache file if it exists.

    Args:
        cache_path: Path to the cache file

    Returns:
        str | None: Cached response if found, None otherwise
    """
    try:
        with open(cache_path, "r") as f:
            return json.load(f)["response"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def get_image_description(
    image_paths: list[str],
    client: OpenAI,
    image_prompt_path: str,
    log_dir: Path,
    detail: str = "high",
) -> str | None:
    """
    Get image descriptions using GPT-4o with caching support.

    Args:
        image_paths: List of paths to images
        client: OpenAI client instance
        image_prompt_path: Path to the image prompt
        log_dir: Path to the log directory
        detail: Level of detail for image description (default: "high")

    Returns:
        str | None: Image description or None if error occurs
    """
    try:
        image_description_prompt = load_prompt(image_prompt_path)

        messages = [
            {"role": "system", "content": image_description_prompt},
            {
                "role": "user",
                "content": "Please describe the morphological features visible in these images.",
            },
        ]

        # Loop over all provided image paths
        # b64 encode them
        # create a message object for each
        # append to the messages list
        image_contents = []
        for path in image_paths:
            if os.path.exists(path):
                mime_type = get_mime_type(path)
                with open(path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                    image_contents.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}",
                                "detail": detail,
                            },
                        }
                    )

        # If there are any image contents, append them to the last message
        if image_contents:
            messages[-1]["content"] = [
                {"type": "text", "text": messages[-1]["content"]},
                *image_contents,
            ]

        # Create hash and check cache
        request_hash = create_request_hash(messages)
        cache_path = get_cache_path(log_dir, request_hash)
        cached_response = load_from_cache(cache_path)

        if cached_response is not None:
            logger.info("Loading cached image description")
            return cached_response

        # Log that we're requesting a new description
        logger.info("Sending request to GPT-4o for image description")
        for msg in messages:
            if isinstance(msg.get("content"), list):
                text_content = next(
                    (item["text"] for item in msg["content"] if item["type"] == "text"),
                    None,
                )
                logger.debug(f"{msg['role'].upper()}: {text_content}")
                logger.debug("(+ images)")
            else:
                logger.debug(f"{msg['role'].upper()}: {msg['content']}")

        # Get new response from GPT
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=False,
            max_tokens=500,
        )

        description = response.choices[0].message.content

        save_to_cache(cache_path, description)

        logger.info("GPT-4o image description: \n\n%s", description)

        return description
    except Exception as e:
        logger.error("Error getting image description: %s", e)
        return None


def save_results(
    results: list,
    log_dir: str,
    explanation: str,
    timestamp: str,
    image_prompt_path: str,
    system_prompt_message_for_prediction_of_score: list,
) -> None:
    """Save all results and related data to the log directory.

    Parameters
    ----------
    results: list
        List of results, each result is a dict with the following keys:
            - image_path: str: Path to the image
            - conclusion: str: Conclusion of the evaluation
            - logit: float: Logit of the image
            - attention: list: Attention of the image
            - binary_logit: int: Binary logit of the image
            - image_description: str: Image description
            - prediction_of_score: str: Prediction of score
    log_dir: str
        Path to the log directory
    explanation: str
        Explanation to be saved
    timestamp: str
        Timestamp of the evaluation
    image_prompt: str
        Image prompt to be saved
    system_prompt_messages: list
        System prompt messages to be saved

    Returns
    -------
    None
    """
    current_dir = Path(log_dir) / timestamp
    current_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Saving results to {current_dir} (explanation.txt, image_prompt.txt, system_prompt_messages.json, results.csv, results.md)"
    )

    with open(current_dir / "explanation.txt", "w") as f:
        f.write(explanation)

    with open(current_dir / "image_prompt.txt", "w") as f:
        f.write(load_prompt(image_prompt_path))

    with open(current_dir / "system_prompt_messages.json", "w") as f:
        json.dump(system_prompt_message_for_prediction_of_score, f, indent=2)

    pd.DataFrame(results).to_csv(current_dir / "results.csv", index=False)
    save_md(results, current_dir)


def save_md(results: list, current_dir: Path) -> None:
    """Generate and save markdown report of results.

    Parameters
    ----------
        results: List of results, each result is a dict with the following keys:
            - image_path: str: Path to the image
            - conclusion: str: Conclusion of the evaluation
            - logit: float: Logit of the image
            - attention: list: Attention of the image
            - binary_logit: int: Binary logit of the image
            - image_description: str: Image description
            - prediction_of_score: str: Prediction of score
        current_dir: Path
            Path to the current directory

    Returns
    -------
    None: Everything is saved to the log directory
    """
    md_content = []

    for result in results:
        md_content.extend(
            [
                f"## Image",
                f"<img src='{result['image_path']}' alt='Provided image' />",
                "",
                f"## Model Output",
                f"- Logit: {result['logit']}",
                f"- Attention: {result['attention']}",
                "",
                f"## Image Description",
                result["image_description"],
                "",
                f"## Prediction of Score",
                result["prediction_of_score"],
                "",
                "---",
                "",
            ]
        )

    with open(current_dir / "results.md", "w") as f:
        f.write("\n".join(md_content))


def create_prediction_prompt(
    image_path: str,
    client: OpenAI,
    explanation_path: str,
    log_dir: Path,
    image_prompt_path: str,
) -> tuple[list[dict], str]:
    """Create and return a prediction prompt for the given image and explanation.

    Parameters
    ----------
    image_path: str
        Path to the image
    client: OpenAI
        OpenAI client
    explanation_path: str
        Path to the explanation

    Returns
    -------
    tuple[list[dict], str]
        A tuple containing the prediction prompt and the image description.
        The prediction prompt is a list of dictionaries, each dictionary is a message.
        The image description is a string.
    """
    image_contents = []
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        image_contents.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{get_mime_type(image_path)};base64,{base64_image}",
                    "detail": "high",
                },
            }
        )

    image_description = get_image_description(
        [image_path], client, image_prompt_path, log_dir
    )
    return [
        {
            "role": "system",
            "content": load_prompt(
                "/Users/y.schirris/oai_server/prompts/workshop_submission/prediction_prompt/predict_with_description_and_explanation.txt"
            ),
        },
        {
            "role": "system",
            "content": f"## Use the following explanation for your prediction: \n{load_prompt(explanation_path)}",
        },
        {
            "role": "user",
            "content": f"## Please predict the score of the following image description based on the explanation.",
        },
        {
            "role": "user",
            "content": f"{image_description}",
        },
    ], image_description
