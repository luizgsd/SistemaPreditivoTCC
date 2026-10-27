# Sistema de Predição para Ataques Cibernéticos Baseado em Inteligência Artificial

Repositório do Trabalho de Conclusão de Curso (TCC), voltado para a detecção automatizada de anomalias em redes corporativas. Este projeto propõe um pipeline robusto que une o processamento de Big Data para extração de logs massivos, uma arquitetura de Machine Learning para detecção e classificação de pacotes anômalos, e uma plataforma SIEM para simulação de um SOC(Centro de Operações de Segurança).

---

## Visão Geral do Projeto
Sistemas tradicionais de detecção de intrusão muitas vezes falham em lidar com o volume e a velocidade das ameaças virtuais. Para resolver isso, este projeto foi dividido em duas etapas:
1. **Engenharia de Dados e Treinamento:** Processamento e limpeza de logs do dataset **CIC-IDS-2017** utilizando `Apache Spark (PySpark)`, seguido pelo treinamento de uma Rede Neural Estática.
2. **Inferência e Monitoramento (Dashboard):** Um painel interativo que recebe tráfego de rede, executa a predição com o modelo de IA congelado e emite alertas simulados no padrão corporativo SIEM.

## Tecnologias utilizadas
* **Linguagem Base:** Python
* **Gerenciamento de Ambiente:** Anaconda para criação do ambiente `tf_env`
* **Big Data e ETL:** Apache Spark (PySpark) integrado via FindSpark
* **Inteligência Artificial:** TensorFlow, Scikit-learn
* **Processamento de Dados:** Pandas e NumPy
* **Front-end:** Streamlit e Plotly

## Estrutura de Diretórios

```text
SistemaPreditivoTCC/
├── data/
│   ├── raw/                       # Dados brutos CIC-IDS-2017 (excluídos do repositório por conta do tamanho)
│   └── processed/                 # Dados pré-processados por meio de ETL (apenas os dados de apresentação estão disponíveis no repositório)
├── src/
│   ├── amostra_etl.py             # Script teste de extração e limpeza utilizando PySpark
│   ├── treinamento_modelo.py      # Script teste de arquitetura e treino da Rede Neural
│   ├── etl_treinamento_completo   # Script de ETL + treinamento da Rede Neural completo utilizando os códigos anteriores como base
│   ├── dashboard_apresentacao.py  # Interface visual e interativa modificada para apresentação
│   └── dashboard.py               # Interface visual e interativa sem modificações
├── modelo_tcc.h5                  # Modelo de IA treinado com o código de treinamento completo no Google Colab
├── .gitignore                     # Regras de exclusão
├── requirements.txt               # Dependências do projeto
└── README.md                      # Documentação oficial do repositório
```

## Como Executar o Projeto Localmente

### 1. Preparação do Ambiente
Certifique-se de ter o [Anaconda](https://www.anaconda.com/) instalado. Clone o repositório e crie o ambiente isolado para garantir a compatibilidade do TensorFlow:
```bash
git clone [https://github.com/luizgsd/SistemaPreditivoTCC.git](https://github.com/luizgsd/SistemaPreditivoTCC.git)
cd SistemaPreditivoTCC
conda create -n tf_env python=3.7.12
conda activate tf_env
```

### 2. Instalação das Dependências
Instale os pacotes listados no `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Inicialização do Dashboard
Para executar a interface visual que utiliza o modelo de IA já treinado e os dados/pacotes modificados para apresentação, rode o seguinte código:
```bash
streamlit run src/dashboard_apresentacao.py
```
Caso queira ver o modelo de IA trabalhando com pacotes de dados não modificados para apresentação (cenário realista), rode o seguinte código:
```bash
streamlit run src/dashboard.py
```

O painel de controle do sistema abrirá no seu navegador padrão, permitindo iniciar o monitoramento, visualizar métricas operacionais interativas e acompanhar as interceptações da rede neural em tempo real.

A diferença entre os dois códigos de inicalização do dashboard é unicamente o modo em que os pacotes são entregues para o modelo de IA analisar:
* No `dashboard_apresentacao.py` ocorre uma "roteirização" e embaralhamento nos pacotes com o intuito de fazer com que mais anomalias possam aparecer e, consequentemente, demonstrar todas as funcionalidades do sistema em um curto período de tempo.
* No `dashboard.py` nenhuma das modificações citadas anteriormente acontecem e os pacotes são entregues na sua ordem original, o que resulta em longos períodos de tráfego normal e ataques esporádicos.