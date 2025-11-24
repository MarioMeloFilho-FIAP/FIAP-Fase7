# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
<a href= "https://www.fiap.com.br/"><img src="assets/logo-fiap.jpg" alt="FIAP - Faculdade de Informática e Admnistração Paulista" border="0" width=40% height=40%></a>
</p>

<br>

# Enterprise Challenge

## Hephaestus

## 👨‍🎓 Integrantes

- <a href="[#](https://www.linkedin.com/in/mariomelofilho)">Carlos Mario Vieira de Melo</a>
- <a href="#">Matheus Cardoso Oliveira Lima</a>
- <a href="https://www.linkedin.com/in/silasfr">Silas Fernandes de Souza Fonseca</a>
- <a href="#">Stephanie Dias dos Santos</a>

## 👩‍🏫 Professores

### Tutor(a)

- <a href="https://www.linkedin.com/company/inova-fusca">Leonardo Ruiz Orabona</a>

### Coordenador(a)

- <a href="https://www.linkedin.com/company/inova-fusca">ANDRÉ GODOI CHIOVATO</a>


# FarmTech Solutions - Fase 7: Sistema Consolidado

![Python](https://img.shields.io/badge/python-3.10--3.12-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

**IA como Fertilizante Digital - Um Novo Agronegócio do Amanhã**

A Fase 7 consolida todas as fases anteriores (1-6) do projeto FarmTech Solutions em um sistema unificado de inteligência agrícola com capacidades avançadas de previsão de séries temporais baseadas em LSTM.

## 🌟 Funcionalidades

- **Sistema Consolidado**: Interface unificada para todos os subsistemas FarmTech
- **Previsão de Séries Temporais com LSTM**: Previsões avançadas para dados de sensores agrícolas
- **Dashboard Interativo**: Visualização em tempo real com Streamlit
- **Suporte Multi-Sensores**: Monitoramento de temperatura, umidade, umidade do solo, intensidade luminosa e pH
- **Arquitetura Modular**: Fácil integração com implementações de fases anteriores
- **Geração de Dados de Exemplo**: Dados sintéticos integrados para testes e demonstração

## 📋 Requisitos

- Python 3.10 a 3.12 (recomendado)
- TensorFlow 2.16+
- Streamlit 1.28+
- Veja `requirements.txt` para lista completa

> **Nota para Python 3.14**: O TensorFlow oficial ainda não suporta Python 3.14. Use Python 3.12 ou instale a versão nightly: `pip install tf-nightly`

## 🚀 Início Rápido

### 1. Configurar Ambiente Virtual

```bash
python3 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 3. Executar o Sistema

```bash
python farmtech_main.py
```

## 📖 Guia de Uso

### Opções do Menu Principal

1. **Verificar Status do Sistema** - Verificar disponibilidade de todos os subsistemas
2. **Gerar Dados de Exemplo** - Criar dados agrícolas sintéticos (Fase2)
3. **Iniciar Coleta de Dados IoT** - Iniciar coleta de dados de sensores (Fase3/4)
4. **Treinar Modelos ML** - Treinar modelos tradicionais de machine learning (Fase4)
5. **Treinar Modelo LSTM** - Treinar modelo de previsão de séries temporais (NOVO)
6. **Abrir Dashboard** - Abrir dashboard consolidado Streamlit
7. **Visão Computacional** - Executar análise de culturas (Fase6)
8. **Informações do Sistema** - Exibir detalhes do sistema
9. **Sair** - Fechar a aplicação

### Treinamento de Modelos LSTM

```bash
# Do menu principal, selecione opção 5
# Ou execute diretamente:
python backend/train_lstm.py
```

O processo de treinamento irá:
- Gerar dados de exemplo se não existirem
- Preparar sequências para entrada LSTM
- Treinar o modelo com early stopping
- Salvar o modelo treinado em `models/saved_models/`
- Gerar gráficos do histórico de treinamento

### Abrindo o Dashboard

```bash
# Do menu principal, selecione opção 6
# Ou execute diretamente:
streamlit run dashboard/farmtech_consolidated_dashboard.py
```

Funcionalidades do dashboard:
- **Visão Geral**: Leituras mais recentes dos sensores e tendências
- **Dados dos Sensores**: Análise detalhada com seleção de intervalo de tempo
- **Previsão de Séries Temporais**: Previsões baseadas em LSTM
- **Status do Sistema**: Disponibilidade de modelos e dados

## 📁 Estrutura do Projeto

```
Fase7/
├── farmtech_main.py              # Ponto de entrada principal
├── requirements.txt              # Dependências Python
├── config/
│   └── system_config.py          # Configuração centralizada
├── models/
│   ├── lstm_predictor.py         # Classe do modelo LSTM
│   ├── time_series_preprocessor.py  # Pré-processamento de dados
│   └── saved_models/             # Modelos treinados (criado em tempo de execução)
├── backend/
│   └── train_lstm.py             # Script de treinamento LSTM
├── dashboard/
│   └── farmtech_consolidated_dashboard.py  # Dashboard Streamlit
├── utils/
│   └── integration_helpers.py    # Utilitários de integração
├── data/                         # Armazenamento de dados (criado em tempo de execução)
├── logs/                         # Arquivos de log (criado em tempo de execução)
└── tests/                        # Testes unitários (a serem implementados)
```

## 🔧 Configuração

Edite `config/system_config.py` para customizar:

- **Caminhos**: Localizações das implementações de fases anteriores
- **Parâmetros LSTM**: Comprimento de sequência, horizonte de previsão, arquitetura do modelo
- **Configuração de Sensores**: Tipos de sensores disponíveis
- **Configurações do Dashboard**: Intervalos de atualização, opções de exibição

## 🧪 Detalhes do Modelo LSTM

### Arquitetura

- **Entrada**: Sequências de leituras de sensores (padrão: 24 passos de tempo)
- **Camadas LSTM**: Configurável (padrão: [64, 32] unidades)
- **Dropout**: Regularização para prevenir overfitting (padrão: 0.2)
- **Saída**: Previsões multi-passo à frente (padrão: 6 passos de tempo)

### Configuração de Treinamento

```python
LSTM_CONFIG = {
    "sequence_length": 24,      # Horas de histórico a usar
    "prediction_horizon": 6,    # Horas a prever à frente
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "lstm_units": [64, 32],
    "dropout_rate": 0.2,
}
```

## 🔗 Integração com Fases Anteriores

### Fase 2: Geração de Dados e Estatísticas
- Geração de dados agrícolas
- Análise estatística com R
- Geração de relatórios Excel

### Fase 3: Coleta de Dados IoT
- Integração de sensores ESP32/Arduino
- Coleta de dados em tempo real
- Visualização básica em dashboard

### Fase 4: Machine Learning
- Treinamento de modelos ML tradicionais
- Dashboard Streamlit
- Avaliação e previsões de modelos

### Fase 6: Visão Computacional
- Análise de imagens de culturas
- Detecção de objetos para monitoramento agrícola

## 📊 Dados de Exemplo

O sistema inclui geração de dados sintéticos para demonstração:

- **Temperatura**: Ciclo diário com variações realistas
- **Umidade**: Correlação inversa com temperatura
- **Umidade do Solo**: Decaimento com eventos de irrigação
- **Intensidade Luminosa**: Ciclo dia/noite
- **Nível de pH**: Estável com pequenas variações

## 🐛 Solução de Problemas

### Erros de Importação

Se encontrar erros de importação, certifique-se de que:
1. O ambiente virtual está ativado
2. Todas as dependências estão instaladas: `pip install -r requirements.txt`
3. Você está executando do diretório Fase7

### Problemas com TensorFlow

Para usuários de Mac M1/M2:
```bash
pip install tensorflow-macos tensorflow-metal
```

Para suporte a GPU em outros sistemas, veja o [guia de instalação do TensorFlow](https://www.tensorflow.org/install).

### Dashboard Não Carrega

Certifique-se de que o Streamlit está instalado:
```bash
pip install streamlit --upgrade
streamlit --version
```

## 📝 Desenvolvimento

### Adicionando Novos Sensores

1. Atualize `SENSOR_COLUMNS` em `config/system_config.py`
2. Modifique a geração de dados em `backend/train_lstm.py`
3. Atualize as visualizações do dashboard

### Estendendo Modelos LSTM

1. Modifique `LSTM_CONFIG` em `config/system_config.py`
2. Ajuste a arquitetura do modelo em `models/lstm_predictor.py`
3. Atualize o pré-processamento em `models/time_series_preprocessor.py`

## 📄 Licença

Este projeto faz parte do programa acadêmico da FIAP.

---

**FarmTech Solutions** - Transformando a agricultura através da inteligência artificial 🌱
