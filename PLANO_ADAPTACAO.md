# Adaptação: sala ocupada ou vazia

## Estado

Dataset escolhido e conferido. A aplicação e os dados versionados ainda são os
originais do professor: o CI inicial verde é a referência anterior à adaptação.
Não considerar esta etapa evidência E1 da aplicação adaptada.

## Problema

Detectar presença de pessoas numa sala a partir de sensores ambientais, para
apoiar o controle de iluminação e climatização.

- Fonte: UCI Room Occupancy Estimation, https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation .
- Licença: CC BY 4.0; preservar atribuição aos autores.
- Base conferida: 10.129 registros e 16 atributos numéricos de sensores.
- Alvo binário: `ocupada = 1` quando `Room_Occupancy_Count > 0`; caso contrário, `0`.
- Retirar a contagem original das entradas para evitar vazamento do alvo.
- Usar data/hora apenas para organizar a separação temporal, não como entrada inicial.
- Não há identificação de pessoas nas variáveis selecionadas.

## Contrato a detalhar antes do treinamento

Priorizar o recall da classe ocupada: falso negativo pode apagar a luz ou
reduzir a climatização enquanto alguém usa a sala. Controlar também a precisão
para evitar desperdício de energia por falsos positivos. Definir e justificar
os limiares antes de avaliar o holdout, sem ajustá-los olhando seus resultados.

## Cuidados com os dados

São medições temporais. Separar treino e holdout respeitando a ordem temporal,
verificar duplicatas/sobreposição e conferir a presença das duas classes nos
conjuntos. Não usar divisão aleatória que misture observações vizinhas sem
justificativa. A normalização deve ser ajustada somente no treino.

## Sequência de trabalho

1. Adaptar conjuntamente os dados, a geração reproduzível e o contrato.
2. Atualizar validações, pré-processamento, métricas, drift, API e cenários.
3. Acrescentar testes de domínio nas quatro camadas e gerar o campeão inicial.
4. Produzir E1–E5 e escrever ADAPTACAO.md, declarando o uso de IA.

O pacote `mlops_demo` será mantido. Não alterar políticas da organização nem
habilitar promoção automática sem decisão específica. A organização bloqueia
atualmente a criação/aprovação de PRs pelo Actions; isso segue pendente.

Cada integrante precisa contribuir com seus próprios commits. A confirmação de
que esta base não foi usada nos labs da disciplina ainda precisa ser registrada.
