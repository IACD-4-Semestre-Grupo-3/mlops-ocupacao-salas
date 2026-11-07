# Fonte dos sensores ambientais

Room Occupancy Estimation, UCI Machine Learning Repository.
Autores: A. Singh e S. Chaudhari (2018), DOI 10.24432/C5P605.
Fonte: https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation
Download: https://archive.ics.uci.edu/static/public/864/room+occupancy+estimation.zip
Licença: Creative Commons Attribution 4.0 International (CC BY 4.0).

O CSV original está preservado byte a byte. Seu SHA-256 está em config.FONTE_SHA256.
A contagem Room_Occupancy_Count gera ocupada=(contagem>0), mas não entra no modelo.
Date/Time servem só à partição temporal. Sensores registram medidas, não áudio ou identidades.
Valores muito repetidos foram deduplicados usando as 16 features arredondadas a seis casas,
antes das divisões; a fonte intacta permite reproduzir e auditar essa escolha.
O relatório data/raw/proveniencia.json contém as contagens reais após a preparação.
