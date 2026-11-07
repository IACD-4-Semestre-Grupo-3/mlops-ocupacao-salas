"""Verifico o contrato da API de ocupação de salas."""

import json

import pytest
from fastapi.testclient import TestClient

from mlops_demo import api, config
from mlops_demo.api import app

pytestmark = pytest.mark.api

cliente = TestClient(app)


@pytest.fixture
def atributos_sensores():
    return [22.0] * 4 + [150.0] * 4 + [0.1] * 4 + [500.0, -2.0, 0.0, 1.0]


def test_medicao_aceita_inclinacao_negativa_de_co2(monkeypatch, atributos_sensores):
    features = [
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
    monkeypatch.setattr(config, "FEATURES", features)
    assert api.Medicao(atributos=atributos_sensores).atributos == atributos_sensores


def test_medicao_usa_quantidade_dinamica_de_features(monkeypatch, atributos_sensores):
    monkeypatch.setattr(config, "FEATURES", config.FEATURES[:-1])
    assert api.Medicao(atributos=atributos_sensores[:-1]).atributos == atributos_sensores[:-1]


def test_predict_rejeita_atributos_ausentes():
    assert cliente.post("/predict", json={}).status_code == 422


def test_health_responde_ok():
    resposta = cliente.get("/health")
    assert resposta.status_code == 200
    assert resposta.json()["status"] == "ok"


def test_predict_devolve_o_formato_combinado(base_holdout, campeao):
    medicao = base_holdout[config.FEATURES].iloc[[0]]
    resposta = cliente.post("/predict", json={"atributos": medicao.iloc[0].tolist()})
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert set(corpo) == {"classe", "probabilidade_ocupada", "versao_codigo"}
    assert corpo["classe"] in {"ocupada", "vazia"}
    assert 0.0 <= corpo["probabilidade_ocupada"] <= 1.0
    coluna = list(campeao.classes_).index(config.CLASSE_POSITIVA)
    esperado = float(campeao.predict_proba(medicao)[0][coluna])
    assert corpo["probabilidade_ocupada"] == round(esperado, 4)


@pytest.mark.parametrize("quantidade", [0, 2, 15, 17])
def test_predict_rejeita_quantidade_errada_de_atributos(quantidade, atributos_sensores):
    valores = (atributos_sensores + [0.0])[:quantidade]
    resposta = cliente.post("/predict", json={"atributos": valores})
    assert resposta.status_code == 422


def test_predict_aceita_inclinacao_negativa_de_co2(base_holdout):
    medicao = base_holdout[config.FEATURES].iloc[0].tolist()
    medicao[config.FEATURES.index("S5_CO2_Slope")] = -2.0
    resposta = cliente.post("/predict", json={"atributos": medicao})
    assert resposta.status_code == 200
    assert resposta.json()["classe"] in {"ocupada", "vazia"}


@pytest.mark.parametrize(
    ("feature", "valor"),
    [
        ("S1_Temp", -20.1),
        ("S1_Temp", 60.1),
        ("S1_Light", -0.1),
        ("S1_Light", 5000.1),
        ("S1_Sound", -0.1),
        ("S1_Sound", 10.1),
        ("S5_CO2", -1.0),
        ("S5_CO2", 249.9),
        ("S5_CO2", 10000.1),
        ("S5_CO2_Slope", -200.1),
        ("S5_CO2_Slope", 200.1),
        ("S6_PIR", -0.1),
        ("S7_PIR", 1.1),
        ("S1_Temp", True),
        ("S5_CO2", "500"),
        ("S6_PIR", None),
    ],
)
def test_predict_rejeita_valores_invalidos(feature, valor, atributos_sensores):
    atributos_sensores[config.FEATURES.index(feature)] = valor
    resposta = cliente.post("/predict", json={"atributos": atributos_sensores})
    assert resposta.status_code == 422


@pytest.mark.parametrize("valor", [float("nan"), float("inf"), -float("inf")])
def test_predict_rejeita_nao_finitos(valor, atributos_sensores):
    atributos_sensores[0] = valor
    resposta = cliente.post(
        "/predict",
        content=json.dumps({"atributos": atributos_sensores}),
        headers={"Content-Type": "application/json"},
    )
    assert resposta.status_code == 422


def test_medicao_aceita_limites_inclusivos():
    limites = {
        "Temp": (-20, 60),
        "Light": (0, 5000),
        "Sound": (0, 10),
        "CO2": (250, 10000),
        "Slope": (-200, 200),
        "PIR": (0, 1),
    }
    for extremo in (0, 1):
        valores = [limites[nome.rsplit("_", 1)[-1]][extremo] for nome in config.FEATURES]
        assert api.Medicao(atributos=valores).atributos == valores


def test_predict_distingue_sala_ocupada_e_vazia(base_holdout, campeao):
    X = base_holdout[config.FEATURES]
    coluna = list(campeao.classes_).index(config.CLASSE_POSITIVA)
    p_ocupada = campeao.predict_proba(X)[:, coluna]
    i_ocupada = int(p_ocupada.argmax())
    i_vazia = int(p_ocupada.argmin())
    for indice, esperado in ((i_ocupada, "ocupada"), (i_vazia, "vazia")):
        resposta = cliente.post("/predict", json={"atributos": X.iloc[indice].tolist()})
        assert resposta.status_code == 200
        assert resposta.json()["classe"] == esperado


def test_predict_rejeita_movimento_pir_fracionario(atributos_sensores):
    atributos_sensores[config.FEATURES.index("S6_PIR")] = 0.5
    assert cliente.post("/predict", json={"atributos": atributos_sensores}).status_code == 422
