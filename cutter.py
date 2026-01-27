import os
import subprocess
import math

# Agora a função aceita 'duration' (padrão 30s se não informar)
def cortar_video(input_path, output_dir, duration=30):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Descobrir a duração total do vídeo original
    cmd_probe = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", input_path
    ]
    result = subprocess.run(cmd_probe, capture_output=True, text=True)
    
    try:
        total_duration = float(result.stdout.strip())
    except:
        total_duration = 0 # Fallback

    print(f"   ⏱️ Duração total: {total_duration}s | Cortes de: {duration}s")

    clips = []
    
    # Lógica simples: Cria cortes sequenciais baseados no tempo escolhido
    # Ex: Se pediu 60s, corta 0-60, 60-120...
    num_cortes = math.floor(total_duration / duration)
    
    # Limita a 5 cortes para não travar seu PC testando vídeos longos
    if num_cortes > 5:
        num_cortes = 5
        print("   ⚠️ Limitando a 5 cortes para teste rápido.")

    for i in range(num_cortes):
        start_time = i * duration
        clip_name = f"clip_{i+1}.mp4"
        clip_path = os.path.join(output_dir, clip_name)

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", input_path,
            "-t", str(duration), # Aqui definimos o tempo exato
            "-c", "copy", # Copia rápido sem re-codificar
            clip_path
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        clips.append(clip_path)

    return clips