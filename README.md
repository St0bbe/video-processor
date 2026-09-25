# Video Processor AI

API em FastAPI para transformar vídeos longos em cortes inteligentes com contexto.

## Fluxo

1. Recebe um vídeo por upload ou URL.
2. Extrai o áudio com FFmpeg.
3. Transcreve com Faster Whisper e mantém timestamps.
4. Envia a transcrição ao Gemini.
5. A IA seleciona os melhores trechos que funcionam sozinhos e preservam contexto.
6. O FFmpeg gera os cortes finais.

## Requisitos

- Python 3.10+
- FFmpeg e FFprobe disponíveis no PATH

Instalação:

```bash
pip install -r requirements.txt
```

Crie um `.env` ou defina as variáveis de ambiente conforme `.env.example`.

> A chave antiga do Gemini que já apareceu no histórico do repositório deve ser revogada e substituída.

## Executar

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="SUA_CHAVE"
uvicorn main:app --reload
```

Abra:

- API: http://127.0.0.1:8000
- Swagger: http://127.0.0.1:8000/docs

## Processar arquivo

Endpoint:

```
POST /process-upload
```

Parâmetros opcionais:

- `max_cortes`: padrão 3
- `min_duracao`: padrão 45s
- `max_duracao`: padrão 180s

## Processar URL

```
POST /process-url
```

Exemplo JSON:

```json
{
  "url": "https://www.youtube.com/watch?v=...",
  "max_cortes": 3,
  "min_duracao": 45,
  "max_duracao": 180
}
```

Os cortes são salvos em `clips/<job_id>/`.

## Observação sobre qualidade

O modelo Whisper padrão é `small`, adequado para CPU. Para maior precisão, defina `WHISPER_MODEL=medium` ou `large-v3`, sabendo que isso exige mais memória e processamento.
