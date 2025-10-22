from dataclasses import dataclass
from openai import OpenAI
from typing import Iterable, Dict, Any, List, Optional
import json
import os
import requests

# ---------- Data Model ----------

@dataclass
class SongEntry:
    rapper_name: str
    song_title: str
    song_year: str | int
    song_lyrics: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SongEntry":
        # Robust keys; tolerate slight variations
        return cls(
            rapper_name=d.get("rapper"),
            song_title=d.get("title"),
            song_year=d.get("year"),
            song_lyrics=d.get("lyrics"),
        )


# ---------- IO: JSONL Reader/Writer ----------

class JSONLReader:
    """Reads one JSON object per line from a local .txt/.jsonl file."""
    def __init__(self, path: str):
        self.path = path

    def read(self) -> Iterable[Dict[str, Any]]:
        with open(self.path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e


class JSONLWriter:
    """Writes one JSON object per line to a local .txt/.jsonl file."""
    def __init__(self, path: str, overwrite: bool = True):
        self.path = path
        if overwrite:
            open(self.path, "w", encoding="utf-8").close()

    def write_many(self, rows: Iterable[Dict[str, Any]]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------- OpenAI Client Wrapper ----------

class LyricsSummarizer:
    """
    Wraps OpenAI client to summarize lyrics in <= 4 sentences.
    Uses the modern Responses API.
    """
    def __init__(self, model: str = "gpt-5-nano", client: Optional[OpenAI] = None):
        self.client = client or OpenAI()
        self.model = model

    def summarize(self, lyrics: str) -> str:
        #prompt = (
        #    "You are a rap lyrics analyst. Read the lyrics and summarize the topic of "
        #    "the lyrics in no more than four sentences.\n\n"
        #    f"--- LYRICS START ---\n{lyrics}\n--- LYRICS END ---"
        #)
        prompt = [
        {
            'role':'system',
            'content': 'You are a rap lyrics analyst. Read the lyrics and summarize the topic of the lyrics in no more than four sentences.'
        },
        {
            'role':'user',
            'content': f'\n{lyrics}\n'
        }
        ]
        # Responses API call
        resp = self.client.responses.create(
            model=self.model,
            input=prompt,
        )
        # The SDK provides output_text for convenience
        return (resp.output_text or "").strip()


# ---------- Prompt Builder ----------

class PromptBuilder:
    """
    Builds a generation prompt from song metadata + summary.
    """
    def build_prompt(self, entry: SongEntry, summary: str) -> str:
        return (
            f"Write a rap song in year {entry.song_year}'s {entry.rapper_name} style. "
            f"The topic is about: {summary}"
        )


# ---------- Lyrics Post-processor ----------

class LyricsPostProcessor:
    """
    Applies requested transformations to the 'completion' (lyrics).
    Step 4: add <song_title> to <lyrics>.
    """
    def add_title_to_lyrics(self, title: str, lyrics: str) -> str:
        title_line = f"{title}".strip()
        if not title_line:
            return lyrics.strip()
        return f"{title_line}\n\n{lyrics.strip()}"

# ---------- Get the lyrics file ----------

def get_input_file(local_path="data/cleaned_lyrics.txt", download_url=None):
    """
    Ensure the input file exists locally. If not, download it from the given URL.

    Parameters:
        local_path (str): path where the file should be saved
        download_url (str): URL to download the file if not present

    Returns:
        str: path to the verified or downloaded input file
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if os.path.exists(local_path):
        print(f"Found local file: {local_path}")
        return local_path

    if not download_url:
        raise ValueError(f"{local_path} not found and no download URL provided.")

    print(f"{local_path} not found. Downloading from {download_url}...")
    try:
        response = requests.get(download_url)
        response.raise_for_status()

        with open(local_path, "wb") as f:
            f.write(response.content)

        print(f"File downloaded and saved to {local_path}")
        return local_path

    except requests.RequestException as e:
        print(f"Failed to download file: {e}")
        raise