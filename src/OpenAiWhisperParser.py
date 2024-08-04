from langchain_community.document_loaders.generic import GenericLoader
import os
import time
from typing import Any, Dict, Iterator, Literal, Optional, Tuple, Union, List

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
import ffmpeg
import openai

class OpenAIWhisperParser(BaseBlobParser):
    """Transcribe and parse audio files.

    Audio transcription is with OpenAI Whisper model.
    Benchmark: 30 mins ~11MB, 60 mins ~22MB
    Only support video below 30 mins for now.

    Args:
        api_key: OpenAI API key
        chunk_duration_threshold: minimum duration of a chunk in seconds
            NOTE: According to the OpenAI API, the chunk duration should be at least 0.1
            seconds. If the chunk duration is less or equal than the threshold,
            it will be skipped.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        chunk_duration_threshold: float = 0.1,
        base_url: Optional[str] = None,
        language: Union[str, None] = None,
        prompt: Union[str, None] = None,
        response_format: Union[
            Literal["json", "text", "srt", "verbose_json", "vtt"], None
        ] = None,
        temperature: Union[float, None] = None,
        timestamp_granularities: Optional[List[str]] = None,
    ):
        self.api_key = api_key
        self.chunk_duration_threshold = chunk_duration_threshold
        self.base_url = (
            base_url if base_url is not None else os.environ.get("OPENAI_API_BASE")
        )
        self.language = language
        self.prompt = prompt
        self.response_format = response_format
        self.temperature = temperature
        self.timestamp_granularities = timestamp_granularities

    @property
    def _create_params(self) -> Dict[str, Any]:
        params = {
            "language": self.language,
            "prompt": self.prompt,
            "response_format": self.response_format,
            "temperature": self.temperature,
            "timestamp_granularities": self.timestamp_granularities,
        }
        return {k: v for k, v in params.items() if v is not None}

    def process_audio_chunk(self, blob_path, start_time, end_time):
        output_path = f"{blob_path}.tmp.mp3"
        ffmpeg.input(blob_path, ss=start_time, t=end_time - start_time).output(
            output_path, format="mp3"
        ).run()
        return output_path

    def transcribe_audio(self, file_path):
        with open(file_path, "rb") as file_obj:
            for attempt in range(1, 4):
                try:
                    transcript = openai.OpenAI(
                        api_key=self.api_key
                    ).audio.transcriptions.create(
                        model="whisper-1", file=file_obj, **self._create_params
                    )
                    return transcript
                except Exception as e:
                    print(f"Attempt {attempt} failed. Exception: {str(e)}")
                    time.sleep(5)
        return None

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        # check if the blob is an audio file, if it is an text file, return the text

        # # Audio file from disk
        # TODO: if the first chunk is already larger than 25MB, we need to to split it into smaller chunks
        chunk_duration = 20
        chunk_duration_second = chunk_duration * 60

        probe = ffmpeg.probe(blob.path)
        duration = float(probe["format"]["duration"])

        detected_lang = None
        result = []
        start_time = 0
        while start_time < duration:
            end_time = min(start_time + chunk_duration_second, duration)
            audio_chunk_path = self.process_audio_chunk(blob.path, start_time, end_time)
            transcript = self.transcribe_audio(audio_chunk_path)
            os.remove(audio_chunk_path)

            if not transcript:
                break  # Exit if transcription fails

            # Set the detected language
            if detected_lang is None:
                print(transcript.to_dict())
                detected_lang = transcript.to_dict()["language"]

            segments = transcript.to_dict()["segments"]
            # adjust the start time and end time of each segment
            for i, _segment in enumerate(segments):
                segments[i]["start"] += start_time
                segments[i]["end"] += start_time

            # Exclude the last segment to avoid overlap, unless it's the last chunk
            if end_time < duration and duration > chunk_duration_second:
                segments = segments[:-1]

            result.extend(segments)

            if duration < chunk_duration_second or end_time == duration:
                break
                # Exit loop if the video is shorter than the chunk duration

            start_time = segments[-1]["end"] + 0.1

        os.remove(blob.path)
        yield Document(
            # update the page content to be the str
            page_content=" ".join([line["text"] for line in result]),
            metadata={
                "segments": result,
                "detected_language": detected_lang,
            },
        )
