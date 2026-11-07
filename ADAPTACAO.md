# Adaptação: sala vazia ou ocupada

## Problema, dados e contrato

Adaptei a base didática de [Wesley Santos](https://github.com/wesleyasantos/ci_cd_ct_mlops)
para estimar sala ocupada, apoiando iluminação e climatização. Não reescrevi os três
pipelines do zero: preservei GitHub Actions, DVC, regressão logística, FastAPI e Docker,
substituindo seus acoplamentos ao domínio original. Não utilizei o dataset Gamma.

Usei [Room Occupancy Estimation (UCI)](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation),
de Adarsh Pal Singh e Sachin Chaudhari (2018), DOI 10.24432/C5P605, licença CC BY 4.0.
A fonte contém 10.129 linhas; selecionei 16 sensores numéricos, sem dados pessoais.
Transformei `Room_Occupancy_Count > 0` em `ocupada=1`; sala vazia é 0.
A contagem original e os timestamps não são entradas do modelo.

O falso negativo é mais caro: poderia desligar a iluminação com pessoas presentes.
Defini antes de avaliar **recall de ocupada ≥ 0,95** (até 5% de falsos negativos entre
ocupações) e **precisão ≥ 0,90**, limitando acionamentos desnecessários.
O portão exige também recall não inferior ao campeão no mesmo conjunto.
São metas didáticas, não certificação para controle de equipamentos.

## Mudanças e reprodução

Preservei o CSV original em `data/source/` e conferi seu SHA-256. Removi 1.305 medições
repetidas, comparando as features arredondadas a seis casas; não houve alvos binários
contraditórios. Separei as 8.824 medições únicas cronologicamente: 1.445 no treino
(22/12/2017), 2.620 no holdout (23/12/2017) e 4.759 na reserva posterior para lotes.
Não há sobreposição de features entre as três partes. Datas seguem o CSV, sem imputação.

Atualizei configuração, geração de dados, limites dos sensores, classe positiva,
métricas e portão. A inclinação de CO₂ pode ser negativa; concentração negativa não.
PIR aceita somente 0 ou 1. A API recebe 16 valores finitos e devolve classe
`ocupada`/`vazia` e `probabilidade_ocupada`. Acrescentei testes em código, dados,
modelo e API. O modelo sempre-ocupada demonstra por que recall sozinho é insuficiente.
Os cenários de drift alteram a luz de lotes da reserva, nunca extraídos do holdout.

```bash
pip install -e ".[dev,ct]"
python -m mlops_demo.data base
python -m mlops_demo.train --campeao
python demo/local.py ci
```

Os comandos de reprodução sobrescrevem dados e campeão locais; executo-os em branch.
Mantive `PROMOCAO_AUTOMATICA=false` e a `main` original, sem merge automático.

## Evidências e estado real

- Base original: [CI verde anterior à adaptação](https://github.com/IACD-4-Semestre-Grupo-3/mlops-ocupacao-salas/actions/runs/42311077472).
- Validação local: 64 testes aprovados; cenário 4 detectou drift e aprovou o candidato;
  cenários 3 e 5 reprovaram por recall e precisão, respectivamente. A promoção do
  reprovado foi bloqueada; a promoção do aprovado foi testada somente em cópia isolada.
- E1: CI da **adaptação na main** depende de revisão e autorização de integração.
  E2: execução vermelha de PR proposital ainda pendente.
- E3/E4: há prova **local**, em `evidencias/validacao-local.json`; faltam os links
  específicos do GitHub Actions para compor as evidências exigidas.
- E5: publicação por tag, pacote público e download/smoke sem login ainda pendentes.
  O Docker daemon não estava acessível na VPS; não apresentei build local inexistente.

O CI de uma feature branch não substitui E1, nem a prova local substitui os cinco runs.
A política organizacional bloqueia Actions de abrir/aprovar PRs; não alterei essa política.

## Limites, contribuições e IA

Um único ambiente controlado não representa outras salas. Sensores exigem calibração;
medições são temporalmente dependentes. Monitoraria disponibilidade, valores impossíveis,
drift, ocupação prevista, recall/precisão com rótulos posteriores e custo dos erros.
O KS é um gatilho de mudança de distribuição, não prova de perda de desempenho.
O holdout histórico mais a metade posterior dos lotes permite comparar os modelos,
mas não é uma avaliação prospectiva independente. Reuso do holdout e múltiplos testes
KS exigem cautela. Não reduzi limiares para aceitar candidatos.

Alvaro Koene Henning conduziu a escolha e a adaptação com assistência do Hermes Agent,
usado em edição de código/documentação, testes, treinamento e verificação do pipeline.
Os commits assistidos registram Alvaro como autor e Hermes como coautor.
Não simulei contribuições dos demais: cada integrante ainda precisa contribuir e
registrar seus próprios commits. O grupo deve confirmar que o dataset não foi usado
nos labs. A entrega D2L permanece pendente das evidências e dessa revisão coletiva.
