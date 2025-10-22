from __future__ import annotations
import os
import time
from typing import  Dict, Any, List, Optional
from utils import JSONLReader, JSONLWriter, LyricsSummarizer, SongEntry, PromptBuilder, LyricsPostProcessor, get_input_file
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log", mode="w", encoding="utf-8"),  # write to file
        logging.StreamHandler()                                     # print to console
    ],
)

class LyricsDataPipeline:
    """
    Orchestrates:
      1) read entries
      2) summarize lyrics via OpenAI
      3) build prompt
      4) add title to lyrics (completion)
      5) emit {prompt, completion}
      6) write output JSONL
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        model: str = "gpt-4o-mini",
        rate_limit_sleep: float = 0.6,  # light throttle; adjust to your limits
    ):
        self.reader = JSONLReader(input_path)
        self.writer = JSONLWriter(output_path, overwrite=True)
        self.summarizer = LyricsSummarizer(model=model)
        self.prompt_builder = PromptBuilder()
        self.post = LyricsPostProcessor()
        self.rate_limit_sleep = rate_limit_sleep

    def run(self, max_items: Optional[int] = None) -> None:
        out_rows: List[Dict[str, Any]] = []

        for i, raw in enumerate(self.reader.read(), 1):
            logging.info(f'Calling API for song NO. {i}')
            if max_items and i > max_items:
                break

            entry = SongEntry.from_dict(raw)
            if not entry.song_lyrics:
                # Skip or handle missing lyrics
                continue

            # (2) summarize one song's lyrics
            summary = self.summarizer.summarize(entry.song_lyrics)
            logging.info(f'Got summary from song NO. {i}')

            # (3) create prompt from metadata and summary
            prompt = self.prompt_builder.build_prompt(entry, summary)

            # (4) add song_title to lyrics
            completion = self.post.add_title_to_lyrics(entry.song_title, entry.song_lyrics)

            # (5) assemble training pair
            out_rows.append({"prompt": prompt, "completion": completion})

            # (rudimentary) rate-limit friendliness
            if self.rate_limit_sleep:
                time.sleep(self.rate_limit_sleep)

            # Periodically flush to disk to avoid large memory usage
            if len(out_rows) >= 100:
                self.writer.write_many(out_rows)
                out_rows.clear()

        # (6) final flush
        if out_rows:
            self.writer.write_many(out_rows)

# ---------- Entry Point ----------

if __name__ == "__main__":
    # Example usage
    INPUT_FILE = get_input_file(
    local_path="data/cleaned_lyrics.txt",
    download_url="https://drive.google.com/file/d/1PqADJhbqqTgyEAXKltgPe6Q0x_zQGLqA/view?usp=drive_link"
)  
    OUTPUT_FILE = "training_pairs.jsonl"

    # Ensure API key is present
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Please set OPENAI_API_KEY in environment.")

    pipeline = LyricsDataPipeline(
        input_path=INPUT_FILE,
        output_path=OUTPUT_FILE,
        model="gpt-5-nano",  
        rate_limit_sleep=0.6,
    )
    pipeline.run()  # optionally: pipeline.run(max_items=50)