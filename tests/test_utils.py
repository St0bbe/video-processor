from ai_editor import formatar_tempo, tempo_para_segundos

def test_formatar_tempo():
    assert formatar_tempo(65) == "00:01:05"
    assert formatar_tempo(3661) == "01:01:01"

def test_tempo_para_segundos():
    assert tempo_para_segundos("01:05") == 65
    assert tempo_para_segundos("01:01:01") == 3661
