"""Configuração central do projeto: caminhos, nomes de colunas e limiares de qualidade."""

from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[2]

# dados
DADOS_BRUTOS = RAIZ / "data" / "raw"
DADOS_LOTES = DADOS_BRUTOS / "lotes"
DADOS_PROCESSADOS = RAIZ / "data" / "processed"
BASE_TREINO = DADOS_BRUTOS / "base_treino.csv"
BASE_HOLDOUT = DADOS_BRUTOS / "base_holdout.csv"
TREINO_PROCESSADO = DADOS_PROCESSADOS / "treino.csv"
AVALIACAO_PROCESSADA = DADOS_PROCESSADOS / "avaliacao.csv"

# modelos e relatórios
MODELO_CAMPEAO = RAIZ / "models" / "champion" / "model.joblib"
METRICAS_CAMPEAO = RAIZ / "models" / "champion" / "metrics.json"
MODELO_CANDIDATO = RAIZ / "models" / "candidate" / "model.joblib"
METRICAS_CANDIDATO = RAIZ / "reports" / "candidate_metrics.json"
RELATORIO_GATE = RAIZ / "reports" / "gate.json"
RESUMO_GATE = RAIZ / "reports" / "gate.md"
RELATORIO_DRIFT = RAIZ / "reports" / "drift.md"
PARAMS = RAIZ / "params.yaml"

# Sensores ambientais; contagem original e timestamp nunca entram no modelo.
FEATURES = [
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
]
ALVO = "ocupada"
CLASSE_POSITIVA = 1
COLUNAS_DRIFT = ["S1_Light", "S2_Light", "S3_Light", "S4_Light"]
# Falso negativo pode apagar a luz numa sala ocupada; falso positivo desperdiça energia.
# Limiares didáticos definidos antes de treinar/avaliar, não certificação de automação.
RECALL_MINIMO = 0.95
PRECISAO_MINIMA = 0.90
TOLERANCIA_CAMPEAO = 0.0
PISO_SANIDADE = 0.85
LIMITE_DRIFT_P = 0.01
MIN_FEATURES_DRIFT = 2
FONTE = RAIZ / "data" / "source" / "Occupancy_Estimation.csv"
FONTE_SHA256 = "c090ee5c94b61762bfec9d22767864490b51030f98cca030f02822468db25d9c"


def carregar_params() -> dict:
    """Lê o params.yaml (hiperparâmetros e semente)."""
    with open(PARAMS, encoding="utf-8") as arquivo:
        return yaml.safe_load(arquivo)
