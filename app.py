
import base64
import io
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from openai import OpenAI


st.set_page_config(
    page_title="Video → Product Insight Extractor",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1280px; padding-top: 2rem;}
      .muted {color:#6b7280; font-size:0.9rem;}
      .tag-seen {background:#dcfce7;color:#166534;padding:3px 8px;border-radius:999px;font-size:.78rem;font-weight:700;}
      .tag-heard {background:#dbeafe;color:#1d4ed8;padding:3px 8px;border-radius:999px;font-size:.78rem;font-weight:700;}
      .tag-inferred {background:#fef3c7;color:#92400e;padding:3px 8px;border-radius:999px;font-size:.78rem;font-weight:700;}
      .tag-confirm {background:#fee2e2;color:#b91c1c;padding:3px 8px;border-radius:999px;font-size:.78rem;font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔎 Video → Product Insight Extractor")
st.caption(
    "Upload 1–6 finished videos of the same product. The app watches the visual timelines, "
    "listens to any speech, and combines the evidence into one commercial product-insight report."
)


@dataclass
class FramePick:
    time_s: float
    frame_index: int
    jpeg_bytes: bytes
    reason: str


def sec_label(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds - (m * 60)
    return f"{m}:{s:04.1f}" if m else f"{s:.1f}s"


def frame_to_jpeg(frame_bgr, quality=82) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        raise ValueError("Could not encode a video frame.")
    return encoded.tobytes()


def data_url(jpeg_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("utf-8")


def analyze_video_locally(video_path: str, target_frames: int) -> Tuple[Dict, List[FramePick]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = total_frames / fps if fps else 0

    scores = []
    prev_small = None
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        small = cv2.resize(frame, (96, 96))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        change = 0.0 if prev_small is None else float(np.mean(cv2.absdiff(gray, prev_small)))
        scores.append((idx, change))
        prev_small = gray
        idx += 1

    cap.release()

    if not scores:
        raise ValueError("No readable frames were found.")

    n = len(scores)
    even_count = max(3, min(target_frames // 2, 6))
    even_indices = set(np.linspace(0, n - 1, even_count, dtype=int).tolist())

    change_indices = []
    min_gap = max(int(fps * 0.45), 1)

    for frame_idx, score in sorted(scores, key=lambda x: x[1], reverse=True):
        if score < 3:
            break
        if all(abs(frame_idx - old) >= min_gap for old in change_indices):
            change_indices.append(frame_idx)
        if len(change_indices) >= max(2, target_frames - even_count):
            break

    selected = sorted(even_indices.union(change_indices))

    if len(selected) > target_frames:
        first, last = selected[0], selected[-1]
        middle = selected[1:-1]
        remaining = max(target_frames - 2, 0)
        keep = [first, last]
        if middle and remaining:
            positions = np.linspace(0, len(middle) - 1, remaining, dtype=int)
            keep += [middle[i] for i in positions]
        selected = sorted(set(keep))

    cap = cv2.VideoCapture(video_path)
    picks = []
    change_set = set(change_indices)

    for frame_idx in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        picks.append(
            FramePick(
                time_s=frame_idx / fps,
                frame_index=frame_idx,
                jpeg_bytes=frame_to_jpeg(frame),
                reason="scene / visual change" if frame_idx in change_set else "timeline coverage",
            )
        )

    cap.release()

    return {
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_s": round(duration, 2),
        "width": width,
        "height": height,
        "selected_frames": len(picks),
    }, picks


def make_contact_sheet(picks: List[FramePick], max_cols: int = 5, thumb_w: int = 190) -> Image.Image:
    if not picks:
        return Image.new("RGB", (800, 300), "white")

    images = []
    for pick in picks:
        img = Image.open(io.BytesIO(pick.jpeg_bytes)).convert("RGB")
        ratio = thumb_w / img.width
        img = img.resize((thumb_w, int(img.height * ratio)))
        images.append((pick, img))

    cell_h = max(img.height for _, img in images) + 32
    cols = min(max_cols, len(images))
    rows = int(np.ceil(len(images) / cols))

    sheet = Image.new("RGB", (cols * thumb_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)

    for i, (pick, img) in enumerate(images):
        x = (i % cols) * thumb_w
        y = (i // cols) * cell_h
        sheet.paste(img, (x, y))
        draw.text((x + 5, y + img.height + 5), sec_label(pick.time_s), fill="black")

    return sheet


def extract_audio(video_path: str) -> str | None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    out.close()

    cmd = [
        ffmpeg, "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", out.name,
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if proc.returncode != 0 or not os.path.exists(out.name) or os.path.getsize(out.name) < 2000:
        try:
            os.unlink(out.name)
        except OSError:
            pass
        return None

    return out.name


def transcribe_audio(client: OpenAI, audio_path: str | None, model: str) -> str:
    if not audio_path:
        return ""
    with open(audio_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(model=model, file=audio_file)
    return getattr(result, "text", "") or ""


def parse_json_loose(text: str) -> Dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def build_prompt(supplied: Dict, videos: List[Dict]) -> str:
    compact_videos = []
    for v in videos:
        compact_videos.append({
            "video_number": v["number"],
            "filename": v["filename"],
            "metadata": v["meta"],
            "transcript": v["transcript"] or "[NO SPEECH TRANSCRIBED / SILENT VIDEO]",
        })

    return f"""
You are a product insight strategist for a high-volume TikTok Shop fashion/content operation.

You are analysing MULTIPLE VIDEOS that are intended to show the SAME PRODUCT.
The chronological frames for each video will follow this instruction.

Your job is NOT to write final hooks yet.
Your job is to combine all the evidence across the videos into the strongest possible product understanding for a separate Hook + CTA Generator.

EVIDENCE LABELS
- OBSERVED = directly visible in one or more videos.
- HEARD = explicitly stated in one or more transcripts.
- SUPPLIED = explicitly provided by the human.
- INFERRED = reasonable marketing interpretation, but NOT a verified factual product claim.
- NEEDS_CONFIRMATION = potentially useful claim that must be verified before being stated as fact.

STRICT CLAIM RULES
Never turn appearance into a factual spec.
Do not invent warmth, comfort, waterproofing, materials, sizes, durability, performance, price, availability or other product claims.
If different videos contradict each other, FLAG THE CONTRADICTION.
If the uploaded videos appear to show different hero products, FLAG THAT.

HUMAN-SUPPLIED PRODUCT INFO
{json.dumps(supplied, indent=2)}

VIDEOS + TRANSCRIPTS
{json.dumps(compact_videos, indent=2)}

RETURN STRICT JSON ONLY:
{{
  "cross_video_check": {{
    "same_product_confidence":"high / medium / low",
    "possible_mismatch_warning":"...",
    "contradictions_or_differences":["..."],
    "what_extra_video_evidence_added":"..."
  }},
  "product_read": {{
    "product_summary":"...",
    "likely_category":"...",
    "combined_visual_story":"...",
    "per_video_summary":[
      {{"video_number":1,"summary":"...","strongest_evidence":"..."}}
    ]
  }},
  "verified_facts":[
    {{"fact":"...","evidence_type":"OBSERVED / HEARD / SUPPLIED","evidence":"...","video_numbers":[1,2]}}
  ],
  "likely_audiences":[
    {{"audience":"...","why":"...","evidence_type":"OBSERVED / HEARD / SUPPLIED / INFERRED"}}
  ],
  "problems_and_solutions":[
    {{"problem":"...","solution_or_outcome":"...","evidence_type":"OBSERVED / HEARD / SUPPLIED / INFERRED","confidence":"high / medium / exploratory"}}
  ],
  "visible_strengths":[
    {{"strength":"...","why_it_matters":"...","evidence_type":"OBSERVED / HEARD / SUPPLIED / INFERRED"}}
  ],
  "desires_and_emotions":[
    {{"desire":"...","angle":"...","evidence_type":"INFERRED / OBSERVED / HEARD / SUPPLIED"}}
  ],
  "occasions":[
    {{"occasion":"...","why_it_fits":"...","evidence_type":"OBSERVED / HEARD / SUPPLIED / INFERRED"}}
  ],
  "possible_objections":[
    {{"objection":"...","how_the_videos_may_answer_it":"...","evidence_type":"OBSERVED / HEARD / SUPPLIED / INFERRED"}}
  ],
  "transformations":[
    {{"before":"...","after":"...","why_useful_for_content":"...","evidence_type":"OBSERVED / HEARD / SUPPLIED / INFERRED"}}
  ],
  "hook_angle_opportunities":[
    {{"angle_type":"identity / problem-solution / styling / transformation / occasion / value / comparison / curiosity / other","angle":"...","why":"..."}}
  ],
  "needs_confirmation":[
    {{"possible_claim":"...","why_confirm":"..."}}
  ],
  "hook_generator_pack": {{
    "product":"...",
    "category":"...",
    "price":"...",
    "audience":"...",
    "problem":"...",
    "solution":"...",
    "strengths":"...",
    "transformation":"...",
    "occasions":"...",
    "objections":"...",
    "emotional_desire":"...",
    "value_or_price_angle":"...",
    "claims_to_avoid":"...",
    "best_hook_angles":"...",
    "extra_context":"..."
  }}
}}
"""


def analyze_with_ai(
    api_key: str,
    vision_model: str,
    transcription_model: str,
    video_records: List[Dict],
    supplied: Dict,
) -> Tuple[Dict, List[Dict]]:
    client = OpenAI(api_key=api_key)

    videos_for_prompt = []
    for record in video_records:
        transcript = transcribe_audio(
            client=client,
            audio_path=record.get("audio_path"),
            model=transcription_model,
        )
        videos_for_prompt.append({
            **record,
            "transcript": transcript,
        })

    prompt = build_prompt(supplied=supplied, videos=videos_for_prompt)
    content = [{"type": "input_text", "text": prompt}]

    for record in videos_for_prompt:
        content.append({
            "type": "input_text",
            "text": f"=== VIDEO {record['number']}: {record['filename']} ===",
        })
        for pick in record["picks"]:
            content.append({
                "type": "input_text",
                "text": f"VIDEO {record['number']} FRAME at {sec_label(pick.time_s)} ({pick.reason})",
            })
            content.append({
                "type": "input_image",
                "image_url": data_url(pick.jpeg_bytes),
                "detail": "low",
            })

    response = client.responses.create(
        model=vision_model,
        input=[{"role": "user", "content": content}],
    )

    return parse_json_loose(response.output_text), videos_for_prompt


def badge(label: str) -> str:
    key = (label or "").upper()
    cls = {
        "OBSERVED": "tag-seen",
        "HEARD": "tag-heard",
        "SUPPLIED": "tag-heard",
        "INFERRED": "tag-inferred",
        "NEEDS_CONFIRMATION": "tag-confirm",
    }.get(key, "tag-inferred")
    return f'<span class="{cls}">{key or "INFERRED"}</span>'


def show_list_section(title: str, rows: List[Dict], main_key: str, second_key: str | None = None):
    st.subheader(title)
    if not rows:
        st.caption("Nothing useful identified here.")
        return

    for row in rows:
        with st.container(border=True):
            st.markdown(badge(row.get("evidence_type", "INFERRED")), unsafe_allow_html=True)
            st.markdown(f"**{row.get(main_key, '')}**")
            if second_key and row.get(second_key):
                st.write(row.get(second_key))


with st.sidebar:
    st.header("Settings")

    default_key = ""
    try:
        default_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        default_key = os.getenv("OPENAI_API_KEY", "")

    api_key = st.text_input(
        "OpenAI API key",
        value=default_key,
        type="password",
        help="Best option: save OPENAI_API_KEY in Streamlit Secrets.",
    )

    vision_model = st.selectbox(
        "Video understanding model",
        ["gpt-5.6-sol", "gpt-6-luna"],
        index=0,
    )

    transcription_model = st.selectbox(
        "Speech transcription model",
        ["gpt-4o-mini-transcribe", "gpt-transcribe", "gpt-4o-transcribe"],
        index=0,
    )

    total_frame_budget = st.slider(
        "Total visual frames sent to AI",
        min_value=18,
        max_value=36,
        value=24,
        step=6,
        help="The app shares this frame budget across all uploaded videos.",
    )

    st.caption(
        "Every video is read locally from start to finish. Only representative frames are sent to the AI."
    )


st.header("1. Upload 1–6 videos of the same product")
uploads = st.file_uploader(
    "MP4 / MOV / M4V",
    type=["mp4", "mov", "m4v"],
    accept_multiple_files=True,
)

if uploads and len(uploads) > 6:
    st.error("Please upload a maximum of 6 videos.")
    uploads = uploads[:6]

video_records = []

if uploads:
    frames_per_video = max(4, total_frame_budget // len(uploads))

    st.caption(
        f"{len(uploads)} video(s) uploaded. The app will use about {frames_per_video} representative frames per video."
    )

    for number, uploaded in enumerate(uploads, start=1):
        cache_key = f"{uploaded.name}-{uploaded.size}-{frames_per_video}"
        session_key = f"video_record_{number}"

        if st.session_state.get(session_key, {}).get("cache_key") != cache_key:
            suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                video_path = tmp.name

            with st.spinner(f"Reading video {number}: {uploaded.name}…"):
                meta, picks = analyze_video_locally(video_path, frames_per_video)
                audio_path = extract_audio(video_path)

            st.session_state[session_key] = {
                "cache_key": cache_key,
                "number": number,
                "filename": uploaded.name,
                "video_path": video_path,
                "meta": meta,
                "picks": picks,
                "audio_path": audio_path,
            }

        record = st.session_state[session_key]
        video_records.append(record)

        with st.expander(f"Video {number}: {uploaded.name}", expanded=(number == 1)):
            st.video(uploaded)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Length", f"{record['meta']['duration_s']:.1f}s")
            c2.metric("Frames read", f"{record['meta']['total_frames']:,}")
            c3.metric("Frames sent", record["meta"]["selected_frames"])
            c4.metric("Audio", "Found" if record.get("audio_path") else "None")
            st.image(make_contact_sheet(record["picks"]), use_container_width=True)


st.header("2. Add anything you already know")
st.caption("Optional. Leave blank if you want the app to infer as much as possible from the videos.")

c1, c2 = st.columns(2)

with c1:
    product_name = st.text_input("Product name (optional)")
    category = st.text_input("Category (optional)")
    price = st.text_input("Price (optional)")
    listing_description = st.text_area("Product/listing description (optional)")

with c2:
    confirmed_facts = st.text_area("Facts you KNOW are true (optional)")
    existing_notes = st.text_area("Your own thoughts (optional)")
    forbidden_claims = st.text_area("Things it must NOT claim (optional)")


st.header("3. Understand the product across all videos")

if not api_key:
    st.info("Add your OpenAI API key in the sidebar.")
elif not uploads:
    st.info("Upload at least one video.")

ready = bool(api_key and uploads and 1 <= len(uploads) <= 6)

if st.button(
    "✨ Analyse all videos together",
    type="primary",
    use_container_width=True,
    disabled=not ready,
):
    supplied = {
        "product_name": product_name.strip() or None,
        "category": category.strip() or None,
        "price": price.strip() or None,
        "listing_description": listing_description.strip() or None,
        "confirmed_facts": confirmed_facts.strip() or None,
        "human_notes": existing_notes.strip() or None,
        "forbidden_or_unverified_claims": forbidden_claims.strip() or None,
    }

    try:
        with st.spinner("Watching every video, listening to speech and combining the evidence…"):
            result, videos_with_transcripts = analyze_with_ai(
                api_key=api_key,
                vision_model=vision_model,
                transcription_model=transcription_model,
                video_records=video_records,
                supplied=supplied,
            )
            st.session_state.insight_result = result
            st.session_state.videos_with_transcripts = videos_with_transcripts
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")


if "insight_result" in st.session_state:
    result = st.session_state.insight_result
    videos_with_transcripts = st.session_state.get("videos_with_transcripts", [])

    st.divider()
    st.header("Combined product understanding")

    check = result.get("cross_video_check", {})
    if check.get("same_product_confidence"):
        st.metric("Same-product confidence", check["same_product_confidence"].upper())

    if check.get("possible_mismatch_warning"):
        warning = check["possible_mismatch_warning"].strip()
        if warning and warning.lower() not in {"none", "n/a", "no"}:
            st.warning(warning)

    if check.get("contradictions_or_differences"):
        with st.expander("Differences / contradictions across videos"):
            for item in check["contradictions_or_differences"]:
                st.write("•", item)

    product_read = result.get("product_read", {})
    st.markdown(f"### {product_read.get('product_summary', 'Product read')}")
    st.write(product_read.get("combined_visual_story", ""))

    if product_read.get("per_video_summary"):
        st.subheader("What each video added")
        st.dataframe(
            pd.DataFrame(product_read["per_video_summary"]),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("What the app heard in each video", expanded=False):
        for record in videos_with_transcripts:
            st.markdown(f"**Video {record['number']}: {record['filename']}**")
            st.write(record.get("transcript") or "No usable speech was transcribed.")

    show_list_section("Verified / grounded facts", result.get("verified_facts", []), "fact", "evidence")
    show_list_section("Who this could be for", result.get("likely_audiences", []), "audience", "why")

    st.subheader("Problems → solutions")
    if result.get("problems_and_solutions"):
        st.dataframe(pd.DataFrame(result["problems_and_solutions"]), use_container_width=True, hide_index=True)

    show_list_section("Strongest product strengths", result.get("visible_strengths", []), "strength", "why_it_matters")
    show_list_section("Desires / emotions", result.get("desires_and_emotions", []), "desire", "angle")
    show_list_section("When / where someone might use it", result.get("occasions", []), "occasion", "why_it_fits")
    show_list_section("Possible shopper objections", result.get("possible_objections", []), "objection", "how_the_videos_may_answer_it")

    st.subheader("Transformations the videos can support")
    if result.get("transformations"):
        st.dataframe(pd.DataFrame(result["transformations"]), use_container_width=True, hide_index=True)

    st.subheader("Best hook-angle opportunities")
    if result.get("hook_angle_opportunities"):
        st.dataframe(pd.DataFrame(result["hook_angle_opportunities"]), use_container_width=True, hide_index=True)

    st.subheader("⚠️ Things to confirm before claiming them")
    confirmations = result.get("needs_confirmation", [])
    if confirmations:
        for row in confirmations:
            with st.container(border=True):
                st.markdown(badge("NEEDS_CONFIRMATION"), unsafe_allow_html=True)
                st.markdown(f"**{row.get('possible_claim', '')}**")
                st.write(row.get("why_confirm", ""))
    else:
        st.success("No obvious unverified product claims were flagged.")

    st.divider()
    st.header("📋 Copy this into the Hook + CTA Generator")

    pack = result.get("hook_generator_pack", {})
    copy_text = "\n".join([
        f"Product: {pack.get('product', '')}",
        f"Category: {pack.get('category', '')}",
        f"Price: {pack.get('price', '')}",
        f"Audience: {pack.get('audience', '')}",
        f"Problem: {pack.get('problem', '')}",
        f"Solution: {pack.get('solution', '')}",
        f"Strengths: {pack.get('strengths', '')}",
        f"Transformation: {pack.get('transformation', '')}",
        f"Occasions: {pack.get('occasions', '')}",
        f"Objections: {pack.get('objections', '')}",
        f"Emotional desire: {pack.get('emotional_desire', '')}",
        f"Price/value angle: {pack.get('value_or_price_angle', '')}",
        f"Claims to avoid: {pack.get('claims_to_avoid', '')}",
        f"Best hook angles: {pack.get('best_hook_angles', '')}",
        f"Extra context: {pack.get('extra_context', '')}",
    ])

    st.code(copy_text, language=None)

    st.download_button(
        "Download Hook Generator pack (.txt)",
        copy_text.encode("utf-8"),
        file_name="hook_generator_pack.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.download_button(
        "Download full report (.json)",
        json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="multi_video_product_insight.json",
        mime="application/json",
        use_container_width=True,
    )
