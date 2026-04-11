"""
TTS Engine — Chatterbox Multilingual com clonagem de voz.

Fluxo:
  1. Converte Gravando.m4a → WAV mono 22050 Hz (via ffmpeg) se necessário
  2. Carrega o modelo Chatterbox-Multilingual 500M (~0.9 GB na primeira vez)
  3. Sintetiza o texto mantendo o timbre da voz de referência
  4. Retorna o caminho para o arquivo WAV gerado

Modelo: ResembleAI/chatterbox-multilingual
Línguas suportadas: pt, en, es, fr, de, it, ja, ko, zh e mais 14
Python: >=3.10 (compatível com 3.12)
"""

import os
import subprocess
import tempfile
import textwrap
from pathlib import Path

from config import REFERENCE_AUDIO

# Áudio de referência para clonagem de voz
REFERENCE_AUDIO_RAW = REFERENCE_AUDIO
# Cache do .wav convertido (mesma pasta, mesmo nome + .wav)
_REFERENCE_WAV_CACHE = Path(REFERENCE_AUDIO_RAW).with_suffix(".ref.wav")

_tts_instance = None


def _ensure_reference_wav() -> str:
    """Converte o áudio de referência para WAV mono 22050 Hz se ainda não convertido."""
    src = Path(REFERENCE_AUDIO_RAW)
    dst = _REFERENCE_WAV_CACHE

    if not src.exists():
        raise FileNotFoundError(
            f"Áudio de referência não encontrado: {src}. "
            "Coloque o arquivo na raiz do projeto ou defina a variável REFERENCE_AUDIO."
        )

    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(src),
                "-ar", "22050",
                "-ac", "1",
                "-acodec", "pcm_s16le",
                str(dst),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return str(dst)


def _get_tts():
    """Carrega o modelo Chatterbox Multilingual (singleton — demora apenas na primeira chamada)."""
    global _tts_instance
    if _tts_instance is None:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # lazy import — pesado
        # device="cpu": RX580 sem suporte ROCm no WSL2; CPU suficiente para 500M params
        _tts_instance = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
    return _tts_instance


def _split_text(text: str, max_chars: int = 220) -> list[str]:
    """
    Divide o texto em partes menores respeitando fronteiras de sentença.
    XTTS v2 tem melhor qualidade com segmentos curtos (~200 chars).
    """
    sentences: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        # quebra por pontuação de fim de sentença
        for sent in paragraph.replace("? ", "?\n").replace("! ", "!\n").replace(". ", ".\n").split("\n"):
            sent = sent.strip()
            if not sent:
                continue
            # se ainda for longa, força quebra por palavra
            if len(sent) <= max_chars:
                sentences.append(sent)
            else:
                for chunk in textwrap.wrap(sent, width=max_chars, break_long_words=False):
                    sentences.append(chunk)
    return sentences


def generate_audio(text: str, language: str = "pt", output_path: str | None = None) -> str:
    """
    Gera um arquivo WAV sintetizando `text` com a voz clonada de REFERENCE_AUDIO.

    Args:
        text: Texto a ser narrado (qualquer tamanho — será segmentado internamente).
        language: Código ISO 639-1 do idioma (padrão: "pt").
        output_path: Caminho do WAV de saída. Se None, cria um arquivo temporário.

    Returns:
        Caminho absoluto do WAV gerado.
    """
    import torch
    import torchaudio as ta

    ref_wav = _ensure_reference_wav()
    model = _get_tts()

    segments = _split_text(text)
    if not segments:
        raise ValueError("Texto vazio após pré-processamento.")

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="study_audio_")
        output_path = tmp.name
        tmp.close()

    sr = model.sr
    silence_frames = int(sr * 0.35)  # 350 ms de pausa entre segmentos

    parts: list[torch.Tensor] = []
    for seg in segments:
        wav = model.generate(
            seg,
            language_id=language,
            audio_prompt_path=ref_wav,
        )
        # wav shape: (1, T) or (T,) — normaliza para (1, T)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        parts.append(wav)
        parts.append(torch.zeros(1, silence_frames))

    combined = torch.cat(parts[:-1], dim=1)  # remove silêncio final
    ta.save(output_path, combined, sr)

    return output_path
