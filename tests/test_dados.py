"""Testes de dados: esquema, valores e vazamento entre treino e teste."""

import pandas as pd
import pytest

from mlops_demo import config
from mlops_demo.data import listar_lotes, validar_dataframe

pytestmark = pytest.mark.dados


def _arquivos_de_dados():
    return [config.BASE_TREINO, config.BASE_HOLDOUT, *listar_lotes()]


@pytest.mark.parametrize("caminho", _arquivos_de_dados(), ids=lambda c: c.name)
def test_arquivo_respeita_esquema_e_valores(caminho):
    problemas = validar_dataframe(pd.read_csv(caminho))
    assert problemas == [], f"{caminho.name}: {problemas}"


def test_base_de_treino_tem_as_duas_classes_em_proporcao_razoavel(base_treino):
    proporcao_ocupada = (base_treino[config.ALVO] == config.CLASSE_POSITIVA).mean()
    assert 0.10 <= proporcao_ocupada <= 0.90


def test_nao_ha_vazamento_entre_treino_e_holdout(base_treino, base_holdout):
    chaves_treino = set(map(tuple, base_treino[config.FEATURES].round(6).to_numpy()))
    chaves_holdout = set(map(tuple, base_holdout[config.FEATURES].round(6).to_numpy()))
    assert chaves_treino.isdisjoint(chaves_holdout)


def test_nao_ha_linhas_duplicadas_na_base_de_treino(base_treino):
    assert not base_treino.duplicated().any()


def test_validador_detecta_nulos_e_valores_negativos(base_treino):
    corrompido = base_treino.copy()
    corrompido.loc[0, "S5_CO2"] = None
    corrompido.loc[1, "S1_Light"] = -5.0
    problemas = validar_dataframe(corrompido)
    assert any("nulos" in p for p in problemas)
    assert any("faixa" in p for p in problemas)


def test_validador_detecta_esquema_diferente(base_treino):
    assert validar_dataframe(base_treino.drop(columns=["S5_CO2"]))


def test_queda_de_co2_e_valida_mas_concentracao_negativa_nao(base_treino):
    sample = base_treino.head(2).copy()
    sample["S5_CO2_Slope"] = -2.0
    assert validar_dataframe(sample) == []
    sample.loc[sample.index[0], "S5_CO2"] = -1
    assert any("S5_CO2" in p for p in validar_dataframe(sample))


def test_reserva_futura_nao_repete_treino_nem_holdout():
    from mlops_demo.data import dividir_dataset

    treino, holdout, reserva = dividir_dataset()
    keys = [
        set(map(tuple, d[config.FEATURES].round(6).to_numpy())) for d in [treino, holdout, reserva]
    ]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])


def test_movimento_pir_fora_de_zero_um_e_recusado(base_treino):
    sample = base_treino.head(2).copy()
    sample.loc[sample.index[0], "S6_PIR"] = 2
    assert any("S6_PIR" in p for p in validar_dataframe(sample))


def test_validacao_rejeita_movimento_pir_fracionario(base_treino):
    sample = base_treino.head(2).copy()
    sample["S6_PIR"] = 0.5
    assert any("S6_PIR" in problema for problema in validar_dataframe(sample))
