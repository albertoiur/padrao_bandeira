## Estratégia de trading baseada em bandeiras (flag)

Este projeto implementa, em Python, uma estratégia de trading baseada no padrão gráfico de bandeira (flag), inspirada no documento *"Estrategia de trading basada en la figura técnica de bandera"*.

### Funcionalidade principal

- Ler uma lista de tickers (`stock600.csv` ou lista manual).
- Descarregar dados diários (`timeframe = 1d`) via `yfinance`.
- Detetar, de forma heurística, o padrão de bandeira alcista:
  - Identificação de um **mastro** (impulso forte de subida).
  - Período curto de **consolidação** (bandeira).
  - **Breakout** acima do topo da bandeira.
- Calcular:
  - **Range** do mastro.
  - **Entry** (preço de fecho da vela de breakout).
  - **Stop Loss** e **Take Profit**:
    - \( TP = Entry + 1 \times Range \)
    - \( SL = Entry - 0.25 \times Range \) (configurável).
- Mostrar uma tabela com os sinais encontrados.
- (Opcional) Exportar os sinais para CSV.

### Instalação

Na diretoria do projeto:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

Certifique-se de que o ficheiro `stock600.csv` está na mesma pasta do script (ou indique o caminho com `--csv`).

### Formato esperado de `stock600.csv`

CSV com uma coluna `Ticket` (ou `Ticker`) com os símbolos, por exemplo:

```text
Ticket,Nome da Empresa
NESN.SW,NESTLE
NOVN.SW,NOVARTIS
...
```

### Como executar

Exemplo a usar `stock600.csv` por omissão:

```bash
python flag_strategy.py
```

Exemplo com lista manual de tickers:

```bash
python flag_strategy.py --tickers "AAPL,MSFT,NVDA,AMZN,TSLA"
```

Exportar os sinais para CSV:

```bash
python flag_strategy.py --output-csv sinais_flag.csv
```

### Timeframes intraday (1 hora e 15 minutos)

- **`flag_strategy_one_hour.py`**: velas de **1h** (período por omissão 60 dias).
- **`flag_strategy_15m.py`**: velas de **15 minutos** (período por omissão 60 dias; o Yahoo costuma limitar intraday a ~60 dias). Por omissão lê **`indices.csv`**.

Exemplo 15m:

```bash
python flag_strategy_15m.py --detection-mode heuristic --plots
python flag_strategy_15m.py --tickers "AAPL,MSFT" --period 30d --lookback 500
```

### Parâmetros configuráveis (linha de comandos)

- **`--csv`**: caminho do CSV com tickers (default: `stock600.csv`).
- **`--tickers`**: lista manual de tickers, separada por vírgulas. Se usada, tem prioridade sobre o CSV.
- **`--pattern-threshold`**: limiar mínimo do *score* interno do padrão. Valores mais altos tornam a deteção mais exigente.
- **`--take-profit-multiplier`**: multiplicador do range para o Take Profit (default: `1.0`).
- **`--stop-loss-multiplier`**: multiplicador do range para o Stop Loss (default: `0.25`).
- **`--output-csv`**: ficheiro onde exportar os sinais encontrados.
- **`--plots`**: gera gráficos PNG do padrão detetado para cada sinal.
- **`--plots-dir`**: diretoria onde guardar os gráficos (default: `plots`).

### Notas sobre a heurística de deteção

A deteção da bandeira é heurística e simplificada:

- Assume que a bandeira ocorre na parte final da série de preços diária.
- Procura:
  - Um **mastro** recente com:
    - Subida mínima (por omissão, ≥ 6%).
    - Proporção mínima de velas bullish (por omissão, ≥ 60%).
  - Uma **bandeira** curta:
    - Range relativamente pequeno em comparação com o range do mastro.
    - Duração inferior à do mastro.
  - Um **breakout**:
    - Última vela com fecho acima do topo da bandeira (com pequeno buffer).
    - Novo máximo em relação ao final do mastro.

Este código foi desenhado para ser facilmente extensível caso queira refinar os critérios, ajustar parâmetros ou adicionar gráficos da figura detetada.
