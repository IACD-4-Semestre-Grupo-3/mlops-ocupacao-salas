"""Sirvo o modelo campeão de ocupação de salas via FastAPI.

Local:   uvicorn mlops_demo.api:app --reload
Docker:  docker run -p 8000:8000 mlops-demo
"""

import json
import math
import os
from numbers import Real
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from mlops_demo import __version__, config
from mlops_demo.data import limites

CAMINHO_MODELO = Path(os.environ.get("MODEL_PATH", config.MODELO_CAMPEAO))
CAMINHO_METRICAS = CAMINHO_MODELO.with_name("metrics.json")

app = FastAPI(title="Ocupação de salas (demo CI/CD/CT)", version=__version__)
_modelo = joblib.load(CAMINHO_MODELO)


class Medicao(BaseModel):
    """Recebo as medições dos sensores na ordem de config.FEATURES."""

    atributos: list[float]

    @field_validator("atributos", mode="before")
    @classmethod
    def validar_atributos(cls, valores: object) -> list[float]:
        """Valido a quantidade, os números finitos e os limites dos sensores."""
        if not isinstance(valores, list) or len(valores) != len(config.FEATURES):
            raise ValueError(f"Espero {len(config.FEATURES)} atributos na ordem de config.FEATURES")
        resultado = []
        for nome, valor in zip(config.FEATURES, valores, strict=True):
            if isinstance(valor, bool) or not isinstance(valor, Real):
                raise ValueError(f"Espero um número real em {nome}, sem booleanos")
            try:
                numero = float(valor)
            except (OverflowError, ValueError) as erro:
                raise ValueError(f"Espero um número finito em {nome}") from erro
            if not math.isfinite(numero):
                raise ValueError(f"Espero um número finito em {nome}")
            # mesmas faixas da validação dos dados de treino (data.limites)
            minimo, maximo = limites(nome)
            if not minimo <= numero <= maximo:
                raise ValueError(f"Espero {nome} entre {minimo} e {maximo}")
            if nome.endswith("_PIR") and numero not in {0, 1}:
                raise ValueError(f"Espero {nome} igual a 0 ou 1")
            resultado.append(numero)
        return resultado


class Predicao(BaseModel):
    classe: Literal["ocupada", "vazia"]
    probabilidade_ocupada: float
    versao_codigo: str


@app.exception_handler(RequestValidationError)
async def erro_de_validacao(_request: Request, erro: RequestValidationError) -> JSONResponse:
    """Devolvo 422 sem tentar serializar NaN ou infinito recebidos na entrada."""
    detalhes = [{chave: item[chave] for chave in ("loc", "msg", "type")} for item in erro.errors()]
    return JSONResponse(status_code=422, content={"detail": detalhes})


@app.get("/health")
def saude() -> dict:
    metricas = {}
    if CAMINHO_METRICAS.exists():
        metricas = json.loads(CAMINHO_METRICAS.read_text(encoding="utf-8"))
    return {"status": "ok", "versao_codigo": __version__, "metricas_do_campeao": metricas}


@app.post("/predict", response_model=Predicao)
def prever(medicao: Medicao) -> Predicao:
    entrada = pd.DataFrame([medicao.atributos], columns=config.FEATURES)
    previsto = int(_modelo.predict(entrada)[0])
    coluna = list(_modelo.classes_).index(config.CLASSE_POSITIVA)
    p_ocupada = float(_modelo.predict_proba(entrada)[0][coluna])
    return Predicao(
        classe="ocupada" if previsto == config.CLASSE_POSITIVA else "vazia",
        probabilidade_ocupada=round(p_ocupada, 4),
        versao_codigo=__version__,
    )
