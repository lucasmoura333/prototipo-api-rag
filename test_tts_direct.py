"""Teste direto do TTS para capturar erros de crash."""
import os, sys, traceback

print("Importando tts_engine...")
try:
    from tts_engine import generate_audio
    print("Import OK")
except Exception:
    traceback.print_exc()
    sys.exit(1)

print("Gerando audio de teste curto...")
try:
    path = generate_audio("Ola mundo, este e um teste.", language="pt")
    print(f"OK: {path} ({os.path.getsize(path)} bytes)")
except Exception:
    traceback.print_exc()
    sys.exit(1)
