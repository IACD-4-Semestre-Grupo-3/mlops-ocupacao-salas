# Ocupação de salas com CI/CD/CT

Adaptei o projeto didático de [Wesley Santos](https://github.com/wesleyasantos/ci_cd_ct_mlops)
para identificar **sala vazia (0) ou ocupada (1)** a partir de sensores ambientais.
Não usei dados nem regras do Gamma. Preservei a estrutura de CI, CT, CD, DVC e FastAPI.
O histórico novo começou com a base original e seu CI verde antes da adaptação.

## Problema e contrato

O protótipo pode apoiar iluminação/climatização. Um falso negativo pode deixar pessoas
no escuro; um falso positivo mantém equipamentos ligados sem necessidade.
Defini antes do treinamento **recall de ocupada ≥ 0,95 e precisão ≥ 0,90**.
O candidato também não pode ter recall inferior ao campeão no mesmo conjunto de avaliação.
São limites didáticos, não autorização para desligar equipamentos reais.

## Dados reproduzíveis

[Room Occupancy Estimation, UCI](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation),
licença **CC BY 4.0**. Preservei a fonte original em `data/source/` com SHA-256 conferido.
São 10.129 registros originais e 16 sensores numéricos. O alvo é
`Room_Occupancy_Count > 0`; contagem, data e hora não entram nas features.

Antes de separar, removi 1.305 repetições de features na precisão de seis casas.
Não encontrei alvos binários contraditórios nessas repetições. Das 8.824 linhas únicas:

- Treino: **1.445**, medidas de 22/12/2017.
- Holdout: **2.620**, medidas de 23/12/2017.
- Reserva para lotes: **4.759**, medidas posteriores, sem usar o holdout para gerar lotes.

As divisões são temporais, não um embaralhamento de medições vizinhas.
`data/raw/proveniencia.json` registra classes, exclusões e checksum.
`data/raw/lotes/` começa vazia. A fonte é pública e não contém identidades ou gravações de áudio.

## Instalar e verificar

Python 3.11 ou 3.12 e Git. A instalação inicial exige internet.

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev,ct]"
python demo/local.py ci
```

O CI valida lint, formatação, dados e testes de código, dados, modelo e API.
Há novos testes nas quatro camadas, inclusive para impedir vazamento do alvo,
aceitar queda de CO₂ e rejeitar PIR fracionário, NaN e sensores impossíveis.

Para reproduzir os CSVs e o campeão inicial a partir da fonte versionada:

```bash
python -m mlops_demo.data base
python -m mlops_demo.train --campeao
```

Esses comandos sobrescrevem as bases/modelo locais; use uma branch de trabalho.

## API

```bash
uvicorn mlops_demo.api:app --host 127.0.0.1 --port 8000
# Em outro terminal com o ambiente ativo:
python demo/smoke_test.py http://localhost:8000
```

`GET /health` informa o estado e as métricas. `POST /predict` recebe
`{"atributos": [...]}` com exatamente 16 números, na ordem:

```text
S1_Temp, S2_Temp, S3_Temp, S4_Temp,
S1_Light, S2_Light, S3_Light, S4_Light,
S1_Sound, S2_Sound, S3_Sound, S4_Sound,
S5_CO2, S5_CO2_Slope, S6_PIR, S7_PIR
```

A resposta contém `classe` (`ocupada`/`vazia`), `probabilidade_ocupada` e `versao_codigo`.
Entradas fora do contrato retornam 422. PIR é binário; a inclinação do CO₂ pode ser negativa.

## CT e cenários de falha

Execute cada cenário numa branch/cópia diferente; não misture alterações e não integre
falhas propositais à `main`.

```bash
python demo/cenarios.py listar
python demo/cenarios.py aplicar 4
python demo/local.py ct
```

1. Remove uma feature: teste de código deve falhar.
2. Insere nulos em CO₂ e luz negativa: validação deve falhar.
3. Regularização excessiva: recall deve reprovar.
4. Lote da reserva com luz multiplicada por 0,75: drift e retreino, portão aprovado.
5. Candidato que sempre prevê ocupada: recall alto, precisão baixa, promoção bloqueada.

O cenário 4 simula mudança de calibração mantendo os rótulos; não representa um sensor
real instalado. Também há mudança natural de distribuição entre datas.
O CT compara candidato e campeão no mesmo holdout histórico acrescido da metade posterior
dos lotes. A primeira metade dos lotes vai para treino, respeitando a ordem das linhas.
Essa comparação não equivale a uma validação prospectiva independente após cada retreino.

## GitHub, entrega e limites

Mantive os workflows `.github/workflows/ci.yml`, `ct.yml` e `cd.yml`.
`PROMOCAO_AUTOMATICA=false`: não autorizei escrita automática na `main`.
A política da organização pode impedir que Actions abra o PR de promoção;
isso não deve ser contornado alterando a segurança global.

O CD continua configurado para tag de versão, com CI, build e smoke antes do GHCR.
Publicação Docker e mudança do pacote para público ainda precisam ser comprovadas.
A imagem pública deve poder ser baixada sem login; repositório público não basta.

[ADAPTACAO.md](ADAPTACAO.md) reúne decisões, riscos, uso de IA e pendências da entrega.
[evidencias/validacao-local.json](evidencias/validacao-local.json) contém testes locais,
não substitutos das cinco execuções exigidas no Actions.
Todos os integrantes precisam fazer contribuições e commits próprios; não simulei autoria.
Antes da entrega, o grupo deve confirmar que este dataset não foi utilizado nos labs.
