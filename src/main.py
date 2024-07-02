import argparse
from langchain_community.document_loaders.generic import GenericLoader
from typing import List
from OpenAiWhisperParser import OpenAIWhisperParser
from YoutubeAudioLoader import YoutubeAudioLoader
from summarize import turn_segment_to_html_summary
import re
import os
import langcodes
from dotenv import load_dotenv

load_dotenv()


def generate_youtube_link(video_id):
    """
    Generates a full YouTube video URL from a video ID.

    Parameters:
    video_id (str): The YouTube video ID.

    Returns:
    str: The full YouTube video URL.
    """
    base_url = "https://www.youtube.com/watch?v="
    return base_url + video_id


def get_youtube_video_id(url):
    """
    Extracts the video ID from a YouTube URL.

    Parameters:
    url (str): The YouTube URL.

    Returns:
    str: The video ID.
    """
    # Regular expression to match YouTube video ID patterns
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)

    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid YouTube URL")


def translate_language_code(code):
    if code == "zh-Hant":
        return "Traditional Chinese"
    try:
        language = langcodes.Language.get(code)
        return language.display_name()
    except:
        return "English"


# import from open ai type
def get_video_summary(urls: List[str], save_dir: str, lang: str):
    loader = GenericLoader(
        YoutubeAudioLoader(urls, save_dir),
        OpenAIWhisperParser(
            api_key=os.getenv("openai_key"),
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        ),
    )

    docs = loader.load()
    # output of the loader is guaranteed to len 1 because we only have one video
    segements = docs[0].metadata["segements"]
    html = turn_segment_to_html_summary(
        segements, urls[0], translate_language_code(lang[0])
    )

    return html


# short, cloud
# python .n8n/main.py --urls "https://www.youtube.com/watch?v=-DPaCgcYAIo"
# long, local
# python ./yt2text/main.py --urls "https://www.youtube.com/watch?v=I6FWyej8e38"
# short, local
# python ./yt2text/main.py --urls "https://www.youtube.com/watch?v=ekuAy3DTfVw"
# python ./yt2text/main.py --urls "https://www.youtube.com/watch?v=gYkKW6bAC4U" --lang "zh-TW"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe YouTube videos.")
    parser.add_argument("--urls", nargs="+", help="YouTube video URLs to transcribe.")
    parser.add_argument("--lang", nargs="+", help="the lang used to write the summary")
    args = parser.parse_args()

    video_id = get_youtube_video_id(args.urls[0])
    video_url = generate_youtube_link(video_id)

    # use url to create a directory to save the audio files
    tmp_dir_prefix = "./tmp/"
    tmp_dir = tmp_dir_prefix + "tmp_" + video_id
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # remove all files in the tmp_dir
    file_list = os.listdir(tmp_dir)
    for file_name in file_list:
        file_path = os.path.join(tmp_dir, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

    html = get_video_summary(args.urls, tmp_dir, args.lang)

    with open(tmp_dir + "/result.html", "w") as f:
        f.write(html)
    print("Summary is saved at " + tmp_dir + "/result.html")
