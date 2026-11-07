"""Teste de fumaça: chama /health e /predict de uma API em execução (sem dependências externas).

Uso: python demo/smoke_test.py http://localhost:8000
"""

import csv
import json
import sys
import urllib.request
from pathlib import Path

URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"
HOLDOUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "base_holdout.csv"


def chamar(caminho: str, corpo: dict | None = None) -> dict:
    dados = json.dumps(corpo).encode() if corpo is not None else None
    requisicao = urllib.request.Request(
        URL + caminho, data=dados, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(requisicao, timeout=10) as resposta:
        return json.loads(resposta.read())


saude = chamar("/health")
assert saude["status"] == "ok", saude
print("health ok | versão:", saude["versao_codigo"])

FEATURES = (
    "S1_Temp",
    "S2_Temp",
    "S3_Temp",
    "S4_Temp",
    "S1_Light",
    "S2_Light",
    "S3_Light",
    "S4_Light",
    "S1_Sound",
    "S2_Sound",
    "S3_Sound",
    "S4_Sound",
    "S5_CO2",
    "S5_CO2_Slope",
    "S6_PIR",
    "S7_PIR",
)

with open(HOLDOUT, encoding="utf-8") as arquivo:
    linha = next(csv.DictReader(arquivo))
assert "ocupada" in linha, "Espero a coluna ocupada na base holdout"
predicao = chamar("/predict", {"atributos": [float(linha[nome]) for nome in FEATURES]})
assert set(predicao) == {"classe", "probabilidade_ocupada", "versao_codigo"}, predicao
assert predicao["classe"] in {"ocupada", "vazia"}, predicao
assert 0.0 <= predicao["probabilidade_ocupada"] <= 1.0, predicao
print("predict ok |", predicao)
