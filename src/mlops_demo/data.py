"""Dados ambientais UCI: esquema de ocupação, separação temporal e lotes didáticos.

CLI: python -m mlops_demo.data base|preparar|validar
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mlops_demo import config

SEMENTE_DADOS = 7


def carregar_dataset_completo() -> pd.DataFrame:
    """Confere a fonte versionada; rótulo vem da contagem, não entra nas features."""
    if hashlib.sha256(config.FONTE.read_bytes()).hexdigest() != config.FONTE_SHA256:
        raise ValueError("Fonte UCI diferente do checksum registrado")
    raw = pd.read_csv(config.FONTE)
    raw["instante"] = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%Y/%m/%d %H:%M:%S")
    raw[config.ALVO] = (raw["Room_Occupancy_Count"] > 0).astype(int)
    raw = raw.sort_values("instante").reset_index(drop=True)
    # A identidade usa a mesma precisão do teste de sobreposição.
    keys = raw[config.FEATURES].round(6)
    check = keys.assign(ocupada=raw[config.ALVO])
    if (check.groupby(config.FEATURES)[config.ALVO].nunique() > 1).any():
        raise ValueError("Medições iguais possuem alvos contraditórios")
    return raw.loc[~keys.duplicated(), [*config.FEATURES, config.ALVO, "instante"]].reset_index(
        drop=True
    )


def dividir_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dia 22/12/2017 treina; dia 23 é holdout; dias posteriores reservam lotes.

    Fronteiras são calendáricas, antes de avaliar modelo. Não misturo medições
    vizinhas aleatoriamente. O holdout serve de referência histórica do portão.
    """
    df = carregar_dataset_completo()
    date = df["instante"].dt.strftime("%Y-%m-%d")
    groups = [df[date == "2017-12-22"], df[date == "2017-12-23"], df[date > "2017-12-23"]]
    if any(group.empty for group in groups):
        raise ValueError("Partição temporal vazia")
    if groups[0]["instante"].max() >= groups[1]["instante"].min():
        raise ValueError("Ordem temporal inválida")
    if groups[1]["instante"].max() >= groups[2]["instante"].min():
        raise ValueError("Reserva anterior ao holdout")
    result = tuple(
        group[[*config.FEATURES, config.ALVO]].reset_index(drop=True) for group in groups
    )
    if any(set(group[config.ALVO]) != {0, 1} for group in result):
        raise ValueError("Cada partição precisa das duas classes")
    return result


def criar_base() -> None:
    treino, holdout, reserva = dividir_dataset()
    config.DADOS_BRUTOS.mkdir(parents=True, exist_ok=True)
    config.DADOS_LOTES.mkdir(parents=True, exist_ok=True)
    treino.to_csv(config.BASE_TREINO, index=False, lineterminator="\n")
    holdout.to_csv(config.BASE_HOLDOUT, index=False, lineterminator="\n")
    print(f"base_treino: {len(treino)} | base_holdout: {len(holdout)} | reserva: {len(reserva)}")
    report = {
        "fonte_sha256": config.FONTE_SHA256,
        "fonte_linhas": 10129,
        "linhas_unicas": len(treino) + len(holdout) + len(reserva),
        "duplicatas_removidas": 10129 - len(treino) - len(holdout) - len(reserva),
        "particao": "treino 2017-12-22; holdout 2017-12-23; reserva posterior",
        "atributos": len(config.FEATURES),
        "classes": {
            name: {str(k): int(v) for k, v in group[config.ALVO].value_counts().items()}
            for name, group in zip(
                ["treino", "holdout", "reserva"], [treino, holdout, reserva], strict=True
            )
        },
    }
    (config.DADOS_BRUTOS / "proveniencia.json").write_text(json.dumps(report, indent=2) + "\n")


def aplicar_drift(df: pd.DataFrame, fator: float) -> pd.DataFrame:
    """Simula recalibração dos quatro sensores de luz, sem mudar o alvo."""
    if not np.isfinite(fator) or fator <= 0:
        raise ValueError("Fator de drift deve ser finito e positivo")
    result = df.copy()
    result[config.COLUNAS_DRIFT] = result[config.COLUNAS_DRIFT] * fator
    return result


def gerar_lote(nome: str, n: int = 120, fator_drift: float = 1.0, semente: int = 1) -> str:
    """Amostra somente a reserva futura; preserva a ordem temporal das linhas."""
    if not nome or Path(nome).name != nome or any(c in nome for c in ["/", "\\", "."]):
        raise ValueError("Nome do lote inválido")
    if n < 4:
        raise ValueError("Lote precisa de pelo menos quatro linhas")
    _, _, reserva = dividir_dataset()
    lote = (
        reserva.sample(n=min(n, len(reserva)), random_state=semente)
        .sort_index()
        .reset_index(drop=True)
    )
    if fator_drift != 1:
        lote = aplicar_drift(lote, fator_drift)
    problems = validar_dataframe(lote)
    if problems:
        raise ValueError("Lote inválido: " + "; ".join(problems))
    config.DADOS_LOTES.mkdir(parents=True, exist_ok=True)
    path = config.DADOS_LOTES / f"{nome}.csv"
    if path.exists():
        raise FileExistsError("Lote já existe; use outro nome para preservar a origem")
    lote.to_csv(path, index=False, lineterminator="\n")
    return str(path)


def limites(coluna: str) -> tuple[float, float]:
    if coluna.endswith("_Temp"):
        return -20, 60
    if coluna.endswith("_Light"):
        return 0, 5000
    if coluna.endswith("_Sound"):
        return 0, 10
    if coluna == "S5_CO2":
        return 250, 10000
    if coluna == "S5_CO2_Slope":
        return -200, 200
    return 0, 1


def validar_dataframe(df: pd.DataFrame) -> list[str]:
    problemas = []
    if list(df.columns) != [*config.FEATURES, config.ALVO]:
        return ["esquema diferente do esperado para sensores e ocupada"]
    if df.empty:
        problemas.append("dataset vazio")
    if df.isna().any().any():
        problemas.append("valores nulos")
    if not all(
        pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]) for c in df
    ):
        return problemas + ["colunas devem ser numéricas, não booleanas ou texto"]
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        problemas.append("valores não finitos")
    if not df[config.ALVO].isin([0, 1]).all():
        problemas.append("alvo fora de {0, 1}")
    for column in config.FEATURES:
        if column.endswith("_PIR") and not df[column].isin([0, 1]).all():
            problemas.append(f"indicador binário inválido em {column}")
        lower, upper = limites(column)
        if not df[column].dropna().between(lower, upper).all():
            problemas.append(f"fora da faixa física/didática em {column}")
    return problemas


def listar_lotes() -> list:
    return sorted(config.DADOS_LOTES.glob("*.csv")) if config.DADOS_LOTES.exists() else []


def preparar() -> None:
    """Treina com a primeira metade de cada lote; avalia a metade posterior.

    O holdout fixo continua na comparação histórica. Não interpreto essa mistura
    como uma nova medição exclusivamente prospectiva de desempenho em produção.
    """
    treino = [pd.read_csv(config.BASE_TREINO)]
    avaliacao = [pd.read_csv(config.BASE_HOLDOUT)]
    for path in listar_lotes():
        lote = pd.read_csv(path)
        if len(lote) < 4 or validar_dataframe(lote):
            raise ValueError("Lote inválido")
        middle = len(lote) // 2
        treino.append(lote.iloc[:middle])
        avaliacao.append(lote.iloc[middle:])
    tr = pd.concat(treino, ignore_index=True)
    av = pd.concat(avaliacao, ignore_index=True)
    key_tr = set(map(tuple, tr[config.FEATURES].round(6).to_numpy()))
    key_av = set(map(tuple, av[config.FEATURES].round(6).to_numpy()))
    if key_tr & key_av:
        raise ValueError("Sobreposição entre treino e avaliação")
    if validar_dataframe(tr) or validar_dataframe(av):
        raise ValueError("Dados inválidos antes do treinamento")
    config.DADOS_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    tr.to_csv(config.TREINO_PROCESSADO, index=False, lineterminator="\n")
    av.to_csv(config.AVALIACAO_PROCESSADA, index=False, lineterminator="\n")
    print(f"lotes usados: {len(listar_lotes())}; treino: {len(tr)}; avaliação: {len(av)}")


def validar_tudo() -> int:
    arquivos = [config.BASE_TREINO, config.BASE_HOLDOUT, *listar_lotes()]
    codigo = 0
    for path in arquivos:
        problemas = validar_dataframe(pd.read_csv(path))
        print(path.name + ": " + ("OK" if not problemas else "; ".join(problemas)))
        codigo |= bool(problemas)
    return int(codigo)


if __name__ == "__main__":
    comando = sys.argv[1] if len(sys.argv) > 1 else ""
    if comando == "base":
        criar_base()
    elif comando == "preparar":
        preparar()
    elif comando == "validar":
        sys.exit(validar_tudo())
    else:
        print(__doc__)
        sys.exit(2)
