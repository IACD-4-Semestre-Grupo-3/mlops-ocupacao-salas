# Adaptação: sala vazia ou ocupada

## Problema, dados e contrato

Adaptação da base didática de [Wesley Santos](https://github.com/wesleyasantos/ci_cd_ct_mlops)
para estimar sala ocupada, apoiando iluminação e climatização. Os três pipelines não foram
reescritos do zero: GitHub Actions, DVC, regressão logística, FastAPI e Docker foram preservados,
substituindo seus acoplamentos ao domínio original. Nada do projeto-base foi reutilizado — ele usa a
base Breast Cancer do scikit-learn — e o *dataset* Gamma não faz parte desta adaptação nem do
contrato de dados. A única fonte é a base pública descrita abaixo.

Fonte: [Room Occupancy Estimation (UCI)](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation),
de Adarsh Pal Singh e Sachin Chaudhari (2018), DOI 10.24432/C5P605, licença CC BY 4.0. A fonte
contém 10.129 linhas; foram selecionados 16 sensores numéricos, sem dados pessoais. A contagem
`Room_Occupancy_Count > 0` virou `ocupada=1`; sala vazia é 0. A contagem original e os timestamps
não são entradas do modelo.

O falso negativo é mais caro: poderia desligar a iluminação com pessoas presentes. Antes de avaliar,
foram definidos **recall de ocupada ≥ 0,95** (até 5% de falsos negativos entre ocupações) e
**precisão ≥ 0,90**, limitando acionamentos desnecessários. O portão exige também recall não inferior
ao campeão no mesmo conjunto. São metas didáticas, não certificação para controle de equipamentos.

## Mudanças e reprodução

O CSV original foi preservado em `data/source/` com o SHA-256 conferido. Foram removidas 1.305
medições repetidas, comparando as features arredondadas a seis casas; não houve alvos binários
contraditórios. As 8.824 medições únicas foram separadas cronologicamente: 1.445 no treino
(22/12/2017), 2.620 no holdout (23/12/2017) e 4.759 na reserva posterior para lotes. Não há
sobreposição de features entre as três partes. As datas seguem o CSV, sem imputação.

Configuração, geração de dados, limites dos sensores, classe positiva, métricas e portão foram
atualizados. A inclinação de CO₂ pode ser negativa; concentração negativa não. PIR aceita somente
0 ou 1. A API recebe 16 valores finitos e devolve classe `ocupada`/`vazia` e
`probabilidade_ocupada`. Há testes em código, dados, modelo e API. O modelo sempre-ocupada demonstra
por que recall sozinho é insuficiente. Os cenários de drift alteram a luz de lotes da reserva, nunca
extraídos do holdout.

```bash
pip install -e ".[dev,ct]"
python -m mlops_demo.data base
python -m mlops_demo.train --campeao
python demo/local.py ci
```

Os comandos de reprodução sobrescrevem dados e campeão locais; devem ser executados em branch.
`PROMOCAO_AUTOMATICA=false` e a `main` original foram mantidos, sem merge automático.

## Evidências no GitHub Actions (E1–E5)

| Evidência | O que comprova | Link |
|---|---|---|
| **E1** | CI verde na `main` depois de integrar a adaptação (commit `1469856a982f`) | [run 42320982008](https://github.com/IACD-4-Semestre-Grupo-3/mlops-ocupacao-salas/actions/runs/42320982008) |
| **E2** | CI vermelho proposital em **Validar os dados brutos**; PR fechado sem merge | [run 42324991267](https://github.com/IACD-4-Semestre-Grupo-3/mlops-ocupacao-salas/actions/runs/42324991267) · [PR #2](https://github.com/IACD-4-Semestre-Grupo-3/mlops-ocupacao-salas/pull/2) |
| **E3** | Lote do cenário 4: drift 16/16, CT retreina e o portão **APROVA** (recall 0,9887 / precisão 0,9975); PR integrado | [run 42393373593](https://github.com/IACD-4-Semestre-Grupo-3/mlops-ocupacao-salas/actions/runs/42393373593) · [PR #3](https://github.com/IACD-4-Semestre-Grupo-3/mlops-ocupacao-salas/pull/3) |
| **E4** | Cenários 5 + 4: o portão **REPROVA** (recall 1,0 / precisão 0,2923) e a promoção não roda; PR fechado sem merge | [run 43529584928](https://github.com/IACD-4-Semestre-Grupo-3/mlops-ocupacao-salas/actions/runs/43529584928) · [PR #5](https://github.com/IACD-4-Semestre-Grupo-3/mlops-ocupacao-salas/pull/5) |
| **E5** | Tag `v1.0.0`, CD verde, pacote público e `docker run` + smoke **sem login** | **pendente** (ver abaixo) |

Detalhes conferidos nos registros de cada execução:

- **E1**: os três jobs do CI (lint/ruff, testes em Python 3.11 e 3.12) passaram no commit
  `1469856a982f`, e o CT de `push` na `main` também ficou verde. É a base de comparação das demais
  evidências; todos os runs seguintes partem dela.
- **E2**: a falha ocorre exatamente no passo `Validar os dados brutos`, antes do treino, nos dois
  jobs de teste, com a mensagem
  `base_treino.csv: valores nulos; valores não finitos; fora da faixa física/didática em S1_Light`.
  Lint e formatação seguem verdes, como esperado: é uma falha de dados, não de infraestrutura.
- **E3**: o drift disparou o gatilho (16 de 16 atributos), o `dvc repro` retreinou e o portão
  aprovou com recall 0,9887 e precisão 0,9975 em 2.680 amostras (2.620 do holdout + 60 da metade de
  avaliação do lote). O PR foi integrado às 17h05 UTC de 07/11. O CT que rodou na `main` depois do merge
  passou e abriu o **PR #4** de promoção do campeão; ele continua aberto e o CI dele aparece como
  **Action required**, porque PR aberto pelo `GITHUB_TOKEN` precisa de "Approve and run workflows".
  Ou seja: **o campeão final ainda não está na `main`**.
- **E4**: o CT falhou no passo `Portão de qualidade (candidato x campeão)` — os passos de promoção
  vêm depois e não executaram. O CI também ficou vermelho em `Testes de modelo`, o que é o esperado
  para um modelo sempre-ocupada. Mesmo em PR o CT só retreina quando o gatilho de drift dispara; por
  isso o E4 combina os cenários 5 e 4 — só o cenário 5 mexe em `params.yaml`, o gatilho não dispara
  e o portão nem chegaria a rodar.
- **E5**: não existe tag `v*.*.*` no repositório (`git ls-remote --tags` vazio), nunca houve execução
  do workflow **CD** e não há pacote publicado no GHCR. Faltam, portanto: tag `v1.0.0` a partir de
  uma `main` verde, CD verde, pacote em visibilidade **Public** e `docker run` + smoke sem login. O
  build Docker local não foi apresentado porque o daemon não estava acessível no ambiente usado.
  A ordem importa: a imagem deve sair depois do merge do PR #4, para carregar o campeão final.

## Achado 1: o drift do E3 é real, não só o cenário 4

Um lote gerado **sem** nenhuma alteração (`python demo/gerar_lote.py lote_sem`, fator 1.0) já marca
drift em **16 de 16 atributos**. A causa não é o cenário 4: a reserva vem dos dias posteriores a
23/12, que caem em feriado, e a sala é usada de outro jeito. Os números estão em
`data/raw/proveniencia.json`:

| Parte | Ocupadas | Total | Proporção |
|---|---|---|---|
| Treino (22/12) | 825 | 1.445 | 57,1% |
| Holdout (23/12) | 782 | 2.620 | 29,8% |
| Reserva (lotes) | 294 | 4.759 | 6,2% |

Como a ocupação move todos os sensores, mudar de 57% para 6% desloca as distribuições marginais — luz,
som, CO₂ e temperatura. Então o drift que aparece no E3 tem **duas causas somadas**: a recalibração
simulada de 0,75 na luz e a mudança real de uso da sala.

O teste KS roda por atributo (16 features, `p < 0,01`, gatilho em 2 ou mais atributos). Ele é um
detector de mudança de distribuição: **não separa** "sensor recalibrado" de "sala usada diferente" e
não prova perda de desempenho. Com 4.759 linhas na reserva, diferenças pequenas já ficam
significativas. Duas consequências práticas:

- A leitura correta do E3 é "o CT reage a mudança de distribuição e o portão decide", não "o cenário 4
  foi detectado".
- Comparar contra uma referência com a mesma proporção de ocupação (ou avaliar separadamente por
  período) reduziria o falso alarme. É trabalho futuro: **não** foi implementado, e os limiares de
  drift não foram afrouxados para o E3 parecer mais limpo do que é.

## Validação local (não substitui E1–E5)

`evidencias/validacao-local.json` registra 64 testes aprovados no commit de origem e os cenários
executados em cópias isoladas: o cenário 4 detectou drift e aprovou o candidato; os cenários 3 e 5
reprovaram por recall (0,8069) e precisão (0,2985), respectivamente. A promoção do reprovado foi
bloqueada e o campeão preservado; a promoção do aprovado foi testada somente em cópia, com
`PROMOCAO_AUTOMATICA=false`.

O CI de uma feature branch não substitui E1, e a prova local não substitui os cinco runs. A política
organizacional bloqueia o Actions de abrir/aprovar PRs e **não foi alterada**; o efeito aparece no
PR #4, cujo CI espera aprovação humana.

## Limites

Um único ambiente controlado não representa outras salas. Sensores exigem calibração; medições são
temporalmente dependentes. O monitoramento real deveria acompanhar disponibilidade, valores
impossíveis, drift, ocupação prevista, recall/precisão com rótulos posteriores e o custo dos erros.

O holdout histórico mais a metade posterior dos lotes permite comparar os modelos, mas não é uma
avaliação prospectiva independente. O reuso do holdout e os múltiplos testes KS exigem cautela, e os
limiares não foram reduzidos para aceitar candidatos. O drift observado tem a ambiguidade descrita no
achado 1: o KS mede mudança, não a causa dela.

## Contribuições do grupo

Situação do repositório em 13/11/2026, conferida em commits, PRs e runs:

| Integrante | Contribuição | Prova |
|---|---|---|
| Alvaro Koene Henning ([@Alvaro-KHg](https://github.com/Alvaro-KHg)) | Importou o projeto-base, adaptou o pipeline (PR #1) e executou **E1** e **E2** | commits de 06–07/11, [PR #1](https://github.com/IACD-4-Semestre-Grupo-3/mlops-ocupacao-salas/pull/1), [PR #2](https://github.com/IACD-4-Semestre-Grupo-3/mlops-ocupacao-salas/pull/2) |
| Gabrieli Cavalcante Boeira ([@gabrieliboeira](https://github.com/gabrieliboeira)) | **E3**: gerou o lote do cenário 4 e integrou o PR #3 | commit `39757d9`, [PR #3](https://github.com/IACD-4-Semestre-Grupo-3/mlops-ocupacao-salas/pull/3) |
| Luiz Feltrin ([@BillNobill](https://github.com/BillNobill)) | Revisão do PR #1: 7 pontos de melhoria e a divisão das evidências | [comentário de revisão](https://github.com/IACD-4-Semestre-Grupo-3/mlops-ocupacao-salas/pull/1#issuecomment-6699684544) |
| Luís Henrique Perez ([@luishenriqueperez](https://github.com/luishenriqueperez)) | **E4** (cenários 5 + 4) e parte dos ajustes do item 3 | [PR #5](https://github.com/IACD-4-Semestre-Grupo-3/mlops-ocupacao-salas/pull/5), [PR #6](https://github.com/IACD-4-Semestre-Grupo-3/mlops-ocupacao-salas/pull/6) |
| Anthony Guilherme Mucelini ([@anthonyMucelini](https://github.com/anthonyMucelini)) | Relatório: links E1–E5, achado 1, voz impessoal do texto e esta seção | PR deste documento |

Nenhuma contribuição foi simulada: cada linha acima corresponde a commit, PR ou run registrado no
repositório, e o relatório não atribui autoria a quem não contribuiu. O **E5** continua pendente — a
revisão sugeriu a tag e o pacote público, mas nada foi executado até aqui.

## Uso de IA

Alvaro Koene Henning conduziu a escolha e a adaptação com assistência do Hermes Agent, usado em
edição de código/documentação, testes, treinamento e verificação do pipeline. Os commits assistidos
registram Alvaro como autor e Hermes como coautor.

Este relatório (links E1–E5, achado 1, voz impessoal e contribuições) foi redigido por Anthony
Guilherme Mucelini com assistência de um agente de IA no OpenCode, que consultou os runs, os PRs e o
`proveniencia.json` pela API do GitHub. Os números citados foram conferidos nos registros do Actions,
não estimados.

O grupo deve confirmar que este dataset não foi usado nos labs. A entrega D2L permanece pendente do
E5 e dessa revisão coletiva.
