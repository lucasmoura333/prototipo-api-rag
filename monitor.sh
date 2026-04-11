#!/bin/bash
# Rodar em terminal separado enquanto o modelo processa.
# Mostra CPU/MEM do Ollama + tokens/s em tempo real.

INTERVAL=2

echo "Monitorando Ollama... (Ctrl+C para parar)"
echo ""

while true; do
    STATS=$(docker stats ollama --no-stream --format "CPU={{.CPUPerc}} MEM={{.MemUsage}}" 2>/dev/null)
    
    # Tokens gerados desde o início (via API)
    RUNNING=$(curl -s http://localhost:11434/api/ps 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    models = d.get('models', [])
    if models:
        m = models[0]
        name = m.get('name','?')
        size_vram = m.get('size_vram', 0) // (1024**2)
        expires = m.get('expires_at','')[:19]
        print(f'{name} | VRAM usada: {size_vram}MB | expira: {expires}')
    else:
        print('nenhum modelo carregado')
except:
    print('ollama offline?')
" 2>/dev/null)

    TIMESTAMP=$(date '+%H:%M:%S')
    printf "\r\033[K[%s] %s | %s" "$TIMESTAMP" "$STATS" "$RUNNING"
    sleep $INTERVAL
done
