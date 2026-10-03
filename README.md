# Multi-Video Product Insight Extractor

Upload **1 to 6 finished videos of the same product**.

The app:

- reads each full visual timeline locally
- extracts each video's audio
- transcribes spoken dialogue
- combines the visual + spoken evidence across all videos
- checks whether the videos appear to show the same hero product
- flags contradictions
- identifies audience, problems/solutions, strengths, objections, occasions, transformations and hook angles
- produces one clean copy/paste pack for the separate Hook + CTA Generator

## Audio

The app genuinely uses the video's audio track for **speech transcription**.

It can therefore use things someone explicitly says in the video, such as:
- price
- available colours
- fit comments
- features
- styling advice

It is not currently designed to deeply interpret background music or sound effects.

## Evidence labels

- OBSERVED
- HEARD
- SUPPLIED
- INFERRED
- NEEDS_CONFIRMATION

## Streamlit deployment

Upload:
- `app.py`
- `requirements.txt`
- `README.md`

Deploy with:
- branch: `main`
- main file: `app.py`

Add this to Streamlit Secrets:

```toml
OPENAI_API_KEY = "your-key-here"
```
