"""Workflows for interacting with Google Drive."""

import re
from pathlib import Path

import requests

from geb.workflows.io import fetch_and_save


def check_quota(text: str, file_path: Path | None = None) -> None:
    """Check if the Google Drive quota has been exceeded.

    Args:
        text: The HTML text to check.
        file_path: The path to the file. If the quota is exceeded, the file will be deleted.
            If None, no file will be deleted.

    Raises:
        ValueError: If the quota has been exceeded.
    """
    if "Google Drive - Quota exceeded" in text:
        if file_path and file_path.exists():
            file_path.unlink()  # remove the incomplete file
        raise ValueError(
            "Too many users have viewed or downloaded this file recently. "
            "Please try accessing the file again later. If the file you are "
            "trying to access is particularly large or is shared with many "
            "people, it may take up to 24 hours to be able to view or "
            "download the file."
        )


def download_from_google_drive(
    file_id: str,
    file_path: Path,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> None:
    """Download a file from Google Drive.

    Handles direct downloads and large file warnings (confirmation pages).

    Args:
        file_id: The Google Drive file ID.
        file_path: The path to save the file to.
        session: An optional requests.Session object.
        timeout: Timeout for requests in seconds.

    Raises:
        RuntimeError: If the download fails or the response is unexpected.
    """
    if file_path.exists():
        return

    file_path.parent.mkdir(parents=True, exist_ok=True)

    if session is None:
        session = requests.Session()

    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    # Probe the URL to see if we get a direct download or a confirmation page
    response = session.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    is_html = "text/html" in content_type

    if is_html:
        # Possible confirmation page or error
        html = response.text
        check_quota(html)

        inputs = dict(re.findall(r'name="([^"]+)" value="([^"]+)"', html))

        if "id" in inputs and "confirm" in inputs:
            action_url_match = re.search(r'form[^>]+action="([^"]+)"', html)

            if not action_url_match:
                raise RuntimeError(
                    f"Could not find form action URL in Google Drive HTML response "
                    f"for file_id='{file_id}'. This may indicate that the Google "
                    "Drive HTML/layout has changed."
                )

            action_url = action_url_match.group(1)

            # Use fetch_and_save for large file download
            fetch_and_save(
                url=action_url,
                file_path=file_path,
                params=inputs,
                session=session,
                timeout=timeout,
            )

            # Check for error page disguised as download
            if file_path.exists() and file_path.stat().st_size < 100_000:
                with open(file_path, "r", errors="ignore") as f:
                    text = f.read()
                if "Google Drive - Quota exceeded" in text:
                    check_quota(text, file_path)
        else:
            # Unknown HTML response
            title_match = re.search(
                r"<title>(.*?)</title>",
                html,
                flags=re.IGNORECASE | re.DOTALL,
            )
            html_title = title_match.group(1).strip() if title_match else ""
            html_snippet = html[:500].strip().replace("\n", " ")

            message_parts = [
                "Unexpected HTML response when downloading file from Google Drive."
            ]
            if html_title:
                message_parts.append(f"HTML title: {html_title}")
            if html_snippet:
                message_parts.append(f"HTML snippet: {html_snippet}")

            raise RuntimeError(" ".join(message_parts))

    else:
        # Direct download detected
        response.close()
        fetch_and_save(
            url=url,
            file_path=file_path,
            session=session,
            timeout=timeout,
        )
