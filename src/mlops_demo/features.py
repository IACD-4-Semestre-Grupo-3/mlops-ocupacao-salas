"""Pré-processamento e construção do pipeline de modelo."""

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops_demo.config import ALVO, FEATURES


def separar_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa atributos e alvo, garantindo as colunas esperadas e a ordem."""
    faltando = [c for c in FEATURES + [ALVO] if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes: {faltando}")
    X = df[FEATURES].copy()
    y = df[ALVO].astype(int)
    return X, y


def construir_pipeline(params: dict) -> Pipeline:
    """Pipeline de treino: padronização + regressão logística."""
    modelo = params["modelo"]
    tipo = modelo.get("tipo", "logistica")
    if tipo not in {"logistica", "sempre_ocupada"}:
        raise ValueError("Tipo de modelo desconhecido")
    classificador = (
        DummyClassifier(strategy="constant", constant=1)
        if tipo == "sempre_ocupada"
        else LogisticRegression(
            C=modelo["C"],
            max_iter=modelo["max_iter"],
            class_weight=modelo["class_weight"],
            random_state=params["semente"],
        )
    )
    return Pipeline(
        [
            ("padronizar", StandardScaler()),
            (
                "classificador",
                classificador,
            ),
        ]
    )
