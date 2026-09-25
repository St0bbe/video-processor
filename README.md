# Video Processor AI

API em FastAPI para analisar vídeos longos e gerar automaticamente os melhores cortes com contexto.

## O que mudou na v3

- upload de vídeo e processamento por URL;
- processamento em background com `job_id`;
- endpoint de status/progresso;
- transcrição com Faster Whisper e timestamps por palavra;
- análise em blocos para vídeos longos;
- segunda etapa de ranking global dos candidatos;
- proteção contra cortes muito parecidos/sobrepostos;
- download direto dos cortes;
- limite configurável de tamanho e duração;
- limpeza automática de arquivos antigos;
- chave do Gemini somente por variável de ambiente.

## Pipeline

1. vídeo entra por upload ou URL;
2. FFmpeg extrai o áudio;
3. Faster Whisper transcreve com timestamps;
4. a transcrição é dividida em blocos com sobreposição;
5. Gemini gera candidatos de cada bloco;
6. candidatos são deduplicados;
7. Gemini faz ranking final do vídeo inteiro;
8. timestamps são ajustados para fronteiras de palavras;
9. FFmpeg gera os cortes finais em MP4.

## Requisitos

- Python 3.10+
- FFmpeg e FFprobe no PATH

Instalação:

```bash
pip install -r requirements.txt
```

No PowerShell:

```powershell
$env:GEMINI_API_KEY="SUA_NOVA_CHAVE"
uvicorn main:app --reload
```

Swagger:

```
http://127.0.0.1:8000/docs
```

## Upload

`POST /process-upload`

Parâmetros:
- `max_cortes`: 1 a 10
- `min_duracao`: mínimo 15s
- `max_duracao`: deve ser maior que o mínimo

A resposta retorna um `job_id`.

## URL

`POST /process-url`

Exemplo:

```json
{
  "url": "https://www.youtube.com/watch?v=...",
  "max_cortes": 3,
  "min_duracao": 45,
  "max_duracao": 180
}
```

## Status

`GET /jobs/{job_id}`

Estados:
- queued
- validating
- transcribing
- analyzing
- cutting
- done
- error

## Download

Depois de `done`:

`GET /jobs/{job_id}/clips/1`

Troque `1` pelo número do corte.

## Configuração

Veja `.env.example`.

Para mais qualidade na transcrição:

```powershell
$env:WHISPER_MODEL="medium"
```

ou, em máquina forte:

```powershell
$env:WHISPER_MODEL="large-v3"
```

## Segurança

A chave antiga do Gemini que apareceu anteriormente no histórico do repositório deve ser revogada. Nunca coloque a nova chave diretamente no código.
