# TODO: _describe_frames() usa Ollama (removido). Refatorar para llama-server
# OpenAI-compatible API quando disponível. _transcribe_audio() (Whisper) funciona normalmente.
from pathlib import Path

from llama_index.core import Document

_OLLAMA_MODEL_VISION = "qwen3.5:9b"
_FRAME_INTERVAL_SECONDS = 30
_WHISPER_MODEL = "base"


def load_videos(video_files: list[Path]) -> list[Document]:
    docs: list[Document] = []
    for path in video_files:
        try:
            transcript_docs = _transcribe_audio(path)
            frame_docs = _describe_frames(path)
            docs.extend(transcript_docs)
            docs.extend(frame_docs)
            print(
                f"[video_reader] {path.name} — OK "
                f"({len(transcript_docs)} transcript, {len(frame_docs)} frames)"
            )
        except Exception as e:
            print(f"[video_reader] Failed to process {path.name}: {e}")
    return docs


def _transcribe_audio(path: Path) -> list[Document]:
    import whisper

    model = whisper.load_model(_WHISPER_MODEL)
    result = model.transcribe(str(path))
    docs: list[Document] = []
    for segment in result["segments"]:
        text = f"[{segment['start']:.0f}s-{segment['end']:.0f}s] {segment['text'].strip()}"
        if text.strip():
            docs.append(
                Document(
                    text=text,
                    metadata={
                        "file_name": path.name,
                        "type": "transcript",
                        "start_time": segment["start"],
                    },
                )
            )
    return docs


def _describe_frames(path: Path) -> list[Document]:
    import base64

    import cv2
    import ollama

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = int(fps * _FRAME_INTERVAL_SECONDS)
    docs: list[Document] = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_b64 = base64.b64encode(buf).decode("utf-8")
            try:
                resp = ollama.chat(
                    model=_OLLAMA_MODEL_VISION,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"Describe this video frame objectively and technically. "
                                f"Time: {timestamp:.0f}s."
                            ),
                            "images": [img_b64],
                        }
                    ],
                )
                description = resp["message"]["content"]
                docs.append(
                    Document(
                        text=f"[Frame at {timestamp:.0f}s] {description}",
                        metadata={
                            "file_name": path.name,
                            "type": "frame_description",
                            "timestamp": timestamp,
                        },
                    )
                )
            except Exception as e:
                print(f"[video_reader] Frame description failed at {timestamp:.0f}s: {e}")
        frame_count += 1

    cap.release()
    return docs
