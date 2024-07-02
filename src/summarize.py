import json
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

task_prompt = """## Role ##
You are a professional transcription summarizer.

## Task ##
Your task is to summarize a long audio transcription chunk. The transcription includes discussions, dialogues, and various topics covered in detail. Your goal is to distill the key points, main ideas, examples and significant details from the transcription into a concise and clear summary in {language}. There is no conclusion in the transcription, do not include any conclusion in your output. DO NOT OUTPUT WITH CODE BLOCK QUOTATION.

"""

chunk_summary_prompt = """
## Example of your output ##
```
[
  {
    "section_title": "Challenges",
    "details": [
      { "text": "Economic barriers: High initial investment costs and competition with traditional energy sources.",
        "start_timestamp_reference_in_second": 60
      },
      { "text": "Environmental impact: Concerns about the ecological footprint of large-scale renewable projects.",
        "start_timestamp_reference_in_second": 120
      },
      { "text": ""Regulatory hurdles: Inconsistent policies and lack of supportive regulations across different regions",
        "start_timestamp_reference_in_second": 130
      }
    ],
  },
  {
    "section_title": "Opportunities",
    "details": [
      { 
        "text": "Technological advancements: Innovations in renewable energy technologies can reduce costs and improve efficiency.",
        "start_timestamp_reference_in_second": 150 
      },
      { 
        "text": "Government incentives: Policies such as tax credits and subsidies can encourage investment in renewable energy.",
        "start_timestamp_reference_in_second": 200 
      },
      { 
        "text": "Public awareness: Increasing awareness about climate change can drive demand for cleaner energy sources.",
        "start_timestamp_reference_in_second": 250 
      }
    ]
  },
{
  "section_title": "Market Trends",
  "details": [
    { 
      "text": "Rising demand: The global demand for renewable energy is growing due to environmental concerns and energy security.",
      "start_timestamp_reference_in_second": 300 
    }
  ]
}
]

```
Here is the transcription you need to summarize:

"""

final_reminder_prompt = """
FINAL REMINDER: 
- DO NOT OUTPUT WITH CODE BLOCK QUOTATION. 
- USE {lang} ONLY FOR THE OUTPUT.
- OUTPUT JSON FORMAT ONLY.
"""


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def turn_segment_to_html_summary(segments: list[str], url: str, lang: str):
    data = segments
    srt = []
    for seg in data:
        srt.append(f"{int(seg['start'])} --> {int(seg['end'])}\n{seg['text']}\n\n")

    # chunk the srt by token, each chunk has at most 16k tokens
    chunked_srt = []
    encoding_name = "o200k_base"
    current_token_count = 0
    current_chunk = []
    for line in srt:
        line_token_count = num_tokens_from_string(line, encoding_name)
        if current_token_count + line_token_count > 32000:
            chunked_srt.append(current_chunk)
            current_chunk = []
            current_token_count = 0
        current_chunk.append(line)
        current_token_count += line_token_count
    if current_chunk:
        chunked_srt.append(current_chunk)
    client = OpenAI(api_key=os.getenv("openai_key"))

    summary = []
    for chunk in chunked_srt:
        transcript = client.chat.completions.create(
            temperature=0.3,
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to output JSON.",
                },
                {
                    "role": "user",
                    "content": task_prompt.format(language=lang)
                    + chunk_summary_prompt
                    + "".join(chunk)
                    + final_reminder_prompt.format(lang=lang),
                },
            ],
        )
        summary.append(transcript.choices[0].message.content)

    aggregated_summary = []
    for s in summary:
        try:
            aggregated_summary.extend(json.loads(s))
        except:
            aggregated_summary.extend(
                {
                    "section_title": "Error",
                    "details": [s + " (Please contact the developer for assistance.)"],
                    "start_timestamp_reference_in_second": 0,
                }
            )
    # for summary in aggregated_summary, convert the output into html format
    html = "<html><body>"
    for summary in aggregated_summary:
        html += f"<h2>{summary['section_title']}</h2>"
        html += "<ul>"
        for detail in summary["details"]:
            html += f"<li>{detail['text']}"
            html += " <a href='{url}&t={second}'>[{second}s]</a>".format(
                url=url, second=detail["start_timestamp_reference_in_second"]
            )
            html += "</li>"
        html += "</ul>"
    html += "</body></html>"

    return html
