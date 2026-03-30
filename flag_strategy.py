import argparse
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import yfinance as yf

# Usar backend não interativo para evitar problemas com threads/Tkinter
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@dataclass
class FlagParams:
    lookback_candles: int = 200
    mast_min_bars: int = 4
    mast_max_bars: int = 20
    min_mast_return_pct: float = 6.0  # movimento mínimo do mastro, em percentagem
    min_bull_ratio: float = 0.6  # percentagem mínima de velas bullish no mastro
    flag_min_bars: int = 4
    flag_max_bars: int = 20  # duração máxima da bandeira (velas)
    max_flag_range_ratio: float = 0.4  # range da bandeira vs range do mastro
    breakout_buffer_pct: float = 0.1  # buffer em percentagem (0.1% = 0.001)
    pattern_threshold: float = 0.0  # limiar adicional opcional sobre o score
    take_profit_multiplier: float = 1.0
    stop_loss_multiplier: float = 0.25
    min_range_pct: float = 0.2  # range mínimo em % do preço de entrada (filtra ruído)


@dataclass
class FlagSignal:
    ticker: str
    entry: float
    stop_loss: float
    take_profit: float
    range_points: float
    mast_start: pd.Timestamp
    mast_end: pd.Timestamp
    flag_start: pd.Timestamp
    flag_end: pd.Timestamp
    breakout_time: pd.Timestamp
    score: float
    # Índices relativos dentro do DataFrame original
    mast_start_idx: int
    mast_end_idx: int
    flag_start_idx: int
    flag_end_idx: int
    breakout_idx: int


def load_tickers_from_csv(path: str) -> List[str]:
    """
    Espera um CSV com pelo menos uma coluna chamada 'Ticket' ou 'Ticker'.
    Usa a coluna 'Ticket' do ficheiro stock600.csv fornecido.
    """
    df = pd.read_csv(path)
    for col in ("Ticker", "Ticket"):
        if col in df.columns:
            tickers = df[col].dropna().astype(str).unique().tolist()
            logging.info("Lidos %d tickers da coluna '%s' em %s", len(tickers), col, path)
            return tickers
    raise ValueError(f"Não foi encontrada coluna 'Ticker' ou 'Ticket' em {path}")


def get_price_data(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    lookback_candles: int = 300,
) -> pd.DataFrame:
    """
    Descarrega dados OHLCV com yfinance e devolve apenas as últimas
    `lookback_candles` velas com colunas standard (Open, High, Low, Close, Volume).
    """
    logging.debug("A descarregar dados de %s (%s, %s)...", ticker, period, interval)
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)

    if df.empty:
        logging.warning("Sem dados descarregados para %s", ticker)
        return df

    # yfinance pode devolver:
    # - colunas simples: ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    # - índice de colunas MultiIndex (quando se pedem vários tickers de uma vez)
    # No teu log, as colunas aparecem como ('Adj Close', 'NESN.SW'), ou seja:
    #   nível 0 = tipo de preço ('Open', 'High', ...)
    #   nível 1 = ticker
    # Normalizamos sempre para colunas simples 'Open', 'High', 'Low', 'Close'.
    if isinstance(df.columns, pd.MultiIndex):
        try:
            levels0 = df.columns.get_level_values(0)
            levels1 = df.columns.get_level_values(1)
            # Caso 1: nível 1 é o ticker (mais comum no teu output)
            if ticker in levels1:
                df = df.xs(ticker, axis=1, level=1)
            # Caso 2: nível 0 é o ticker
            elif ticker in levels0:
                df = df.xs(ticker, axis=1, level=0)
        except Exception:  # noqa: BLE001
            logging.warning("Não foi possível normalizar colunas MultiIndex para %s", ticker)

    required_cols = ["Open", "High", "Low", "Close"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logging.warning(
            "Ticker %s sem colunas OHLC completas (em falta: %s). Ticker será ignorado.",
            ticker,
            ", ".join(missing),
        )
        return pd.DataFrame()

    df = df.tail(lookback_candles).copy()
    try:
        df.dropna(subset=required_cols, inplace=True)
    except KeyError:
        # Em alguns casos raros, as colunas podem continuar em formato não standard.
        logging.warning(
            "Não foi possível aplicar dropna nas colunas OHLC para %s. "
            "Ticker será ignorado. Colunas atuais: %s",
            ticker,
            list(df.columns),
        )
        return pd.DataFrame()

    if df.empty:
        logging.warning("Após limpeza de NaN, sem dados suficientes para %s", ticker)
        return df
    return df


# Matriz de pesos para padrão bandeira alcista (referência do documento/vídeo).
# Linhas = níveis de preço (0 = preço alto, 9 = preço baixo). Colunas = tempo (0 = mais antigo, 9 = breakout).
# 0 = preço deve cair nesta célula (caminho ideal). -1 = desvio aceitável. < -1 = penalização.
WEIGHT_MATRIX_BULL_FLAG = np.array(
    [
        [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0],   # linha 0: preço alto (breakout)
        [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, -1, -1],
        [-2, 0, 0, 0, 0, 0, 0, 0, -1, -1],
        [-2, 0, 0, 0, 0, 0, 0, -1, -2, -2],
        [-2, 0, 0, 0, 0, 0, -1, -1, -2, -2],
        [-2, 0, 0, 0, 0, -1, -1, -2, -2, -2],
        [-2, 0, 0, 0, -1, -1, -2, -2, -2, -2],
        [0, 0, 0, -1, -1, -2, -2, -2, -2, -2],
        [0, 0, -1, -2, -2, -2, -2, -2, -2, -2],   # linha 9: preço baixo (início mastro)
    ],
    dtype=float,
)

# Janela fixa para a matriz: 10 velas (3 mastro + 5 bandeira + 2 breakout)
WEIGHT_MATRIX_WINDOW = 10
WEIGHT_MATRIX_MAST_COLS = 3
WEIGHT_MATRIX_FLAG_COLS = 5
WEIGHT_MATRIX_BREAKOUT_COLS = 2


def detect_flag_pattern_weight_matrix(
    df: pd.DataFrame,
    params: FlagParams,
    min_score: float = -5.0,
) -> Optional[Tuple[int, int, int, int, int, float]]:
    """
    Deteção do padrão bandeira através da matriz de pesos.
    Sobrepondo a matriz à janela de preços, o caminho do preço (fechos) é
    convertido em células; soma-se o peso de cada célula visitada.
    Se a soma >= min_score, considera-se bandeira válida.
    Retorna o mesmo formato que detect_flag_pattern (índices + score).
    """
    n = len(df)
    if n < WEIGHT_MATRIX_WINDOW:
        return None

    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values

    # Últimas WEIGHT_MATRIX_WINDOW velas
    start_idx = n - WEIGHT_MATRIX_WINDOW
    window_closes = closes[start_idx : start_idx + WEIGHT_MATRIX_WINDOW]
    window_high = float(highs[start_idx:].max())
    window_low = float(lows[start_idx:].min())
    price_range = window_high - window_low
    if price_range <= 0:
        return None

    n_rows, n_cols = WEIGHT_MATRIX_BULL_FLAG.shape
    # Mapear cada vela (coluna) e preço de fecho → linha (0 = alto, n_rows-1 = baixo)
    score = 0.0
    for col, close in enumerate(window_closes):
        if col >= n_cols:
            break
        # row 0 = preço alto, row 9 = preço baixo
        r = (close - window_low) / price_range
        r = max(0.0, min(1.0, r))
        row = int(round((1.0 - r) * (n_rows - 1)))
        row = max(0, min(n_rows - 1, row))
        score += WEIGHT_MATRIX_BULL_FLAG[row, col]

    if score < min_score:
        return None

    # Índices no DataFrame: mastro 3 velas, bandeira 5, breakout 2
    mast_start_idx = start_idx
    mast_end_idx = start_idx + WEIGHT_MATRIX_MAST_COLS - 1
    flag_start_idx = mast_end_idx + 1
    flag_end_idx = flag_start_idx + WEIGHT_MATRIX_FLAG_COLS - 1
    breakout_idx = n - 1

    # Range = altura do mastro (mínimo no mastro até fecho no fim do mastro)
    mast_closes = closes[mast_start_idx : mast_end_idx + 1]
    mast_start_price = float(np.min(mast_closes))
    mast_end_price = float(closes[mast_end_idx])
    range_pts = mast_end_price - mast_start_price
    if range_pts <= 0:
        return None

    # Breakout: fecho da última vela deve estar acima do topo da bandeira
    flag_high = float(highs[flag_start_idx : flag_end_idx + 1].max())
    if closes[breakout_idx] <= flag_high:
        return None

    # Score para output: escala 0–20 (matriz soma ~ -5 a 0 para padrões aceites)
    output_score = max(0.0, min(20.0, (5.0 + score) * 4.0))

    return (
        mast_start_idx,
        mast_end_idx,
        flag_start_idx,
        flag_end_idx,
        breakout_idx,
        output_score,
    )


def _compute_flag_score(
    mast_return_pct: float,
    flag_range_ratio: float,
    bull_ratio: float,
) -> float:
    """
    Define um score simples para o padrão de bandeira.
    Valores mais altos indicam um padrão mais forte.
    """
    # Evitar divisões excessivas para ranges muito pequenos
    stability_factor = max(0.3, 1.0 - flag_range_ratio)
    score = mast_return_pct * stability_factor * (0.5 + 0.5 * bull_ratio)
    return score


def detect_flag_pattern(df: pd.DataFrame, params: FlagParams) -> Optional[Tuple[int, int, int, int, int, float]]:
    """
    Tenta detetar um padrão de bandeira alcista na parte final da série.

    Retorna:
        (mast_start_idx, mast_end_idx, flag_start_idx, flag_end_idx, breakout_idx, score)
    ou None se não encontrar padrão.
    """
    if df.empty or len(df) < (params.mast_min_bars + params.flag_min_bars + 2):
        return None

    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)

    # Por simplicidade, assumimos que a bandeira está no final da série,
    # e a última vela é potencial vela de breakout.
    breakout_idx = n - 1

    # Tentamos vários tamanhos de bandeira dentro [flag_min_bars, flag_max_bars]
    best_score = -np.inf
    best_pattern: Optional[Tuple[int, int, int, int, int, float]] = None

    for flag_bars in range(params.flag_min_bars, params.flag_max_bars + 1):
        flag_end_idx = breakout_idx - 1
        flag_start_idx = flag_end_idx - flag_bars + 1
        if flag_start_idx <= 0:
            continue

        # Janela do mastro: antes da bandeira, limitada por mast_max_bars
        mast_end_idx = flag_start_idx - 1
        mast_start_idx = max(0, mast_end_idx - params.mast_max_bars + 1)
        mast_len = mast_end_idx - mast_start_idx + 1
        if mast_len < params.mast_min_bars:
            continue

        mast_closes = closes[mast_start_idx : mast_end_idx + 1]
        mast_opens = df["Open"].values[mast_start_idx : mast_end_idx + 1]

        # Encontrar mínimo local como início do mastro
        local_min_idx = mast_start_idx + int(np.argmin(mast_closes))
        if local_min_idx >= mast_end_idx:
            continue

        mast_start_price = closes[local_min_idx]
        mast_end_price = closes[mast_end_idx]
        mast_return_pct = (mast_end_price / mast_start_price - 1.0) * 100.0

        if mast_return_pct < params.min_mast_return_pct:
            continue

        # Percentagem de velas bullish no mastro
        bull_candles = (mast_closes > mast_opens).sum()
        bull_ratio = bull_candles / float(len(mast_closes))
        if bull_ratio < params.min_bull_ratio:
            continue

        # Características da bandeira
        flag_closes = closes[flag_start_idx : flag_end_idx + 1]
        flag_highs = highs[flag_start_idx : flag_end_idx + 1]
        flag_lows = lows[flag_start_idx : flag_end_idx + 1]

        mast_range = mast_end_price - mast_start_price
        if mast_range <= 0:
            continue

        flag_range = float(flag_highs.max() - flag_lows.min())
        flag_range_ratio = flag_range / mast_range
        if flag_range_ratio > params.max_flag_range_ratio:
            continue

        # Breakout: último fecho acima do máximo da bandeira e do fim do mastro
        flag_high = float(flag_highs.max())
        breakout_price = closes[breakout_idx]
        buffer = flag_high * (params.breakout_buffer_pct / 100.0)
        min_breakout_price = flag_high + buffer

        if breakout_price < min_breakout_price:
            continue
        if breakout_price < mast_end_price:
            continue

        score = _compute_flag_score(
            mast_return_pct=mast_return_pct,
            flag_range_ratio=flag_range_ratio,
            bull_ratio=bull_ratio,
        )

        if score < params.pattern_threshold:
            continue

        if score > best_score:
            best_score = score
            best_pattern = (
                local_min_idx,
                mast_end_idx,
                flag_start_idx,
                flag_end_idx,
                breakout_idx,
                score,
            )

    return best_pattern


def calculate_trade_levels(
    entry: float,
    range_points: float,
    take_profit_multiplier: float,
    stop_loss_multiplier: float,
) -> Tuple[float, float]:
    """
    Calcula níveis de TP e SL de acordo com:
        TP = Entry + take_profit_multiplier * Range
        SL = Entry - stop_loss_multiplier * Range
    """
    tp = entry + take_profit_multiplier * range_points
    sl = entry - stop_loss_multiplier * range_points
    return sl, tp


def analyze_ticker(
    ticker: str,
    params: FlagParams,
    detection_mode: str = "heuristic",
) -> Optional[FlagSignal]:
    """
    Orquestra a análise de um ticker:
        - descarrega dados
        - tenta detetar padrão de bandeira (heurística ou matriz de pesos)
        - calcula níveis de entrada, SL e TP
    """
    df = get_price_data(
        ticker=ticker,
        period="1y",
        interval="1d",
        lookback_candles=params.lookback_candles,
    )
    if df.empty:
        return None

    if detection_mode == "weight_matrix":
        pattern = detect_flag_pattern_weight_matrix(df, params)
    else:
        pattern = detect_flag_pattern(df, params)
    if pattern is None:
        return None

    mast_start_idx, mast_end_idx, flag_start_idx, flag_end_idx, breakout_idx, score = pattern

    mast_end_price = float(df["Close"].iloc[mast_end_idx])
    mast_start_price = float(df["Close"].iloc[mast_start_idx : mast_end_idx + 1].min())
    range_points = mast_end_price - mast_start_price

    entry_price = float(df["Close"].iloc[breakout_idx])

    # Filtrar sinais com range insignificante (ruído, ex. 0.03 pontos)
    min_range = entry_price * (params.min_range_pct / 100.0)
    if range_points < min_range:
        return None

    sl, tp = calculate_trade_levels(
        entry=entry_price,
        range_points=range_points,
        take_profit_multiplier=params.take_profit_multiplier,
        stop_loss_multiplier=params.stop_loss_multiplier,
    )

    return FlagSignal(
        ticker=ticker,
        entry=entry_price,
        stop_loss=sl,
        take_profit=tp,
        range_points=range_points,
        mast_start=df.index[mast_start_idx],
        mast_end=df.index[mast_end_idx],
        flag_start=df.index[flag_start_idx],
        flag_end=df.index[flag_end_idx],
        breakout_time=df.index[breakout_idx],
        score=score,
        mast_start_idx=mast_start_idx,
        mast_end_idx=mast_end_idx,
        flag_start_idx=flag_start_idx,
        flag_end_idx=flag_end_idx,
        breakout_idx=breakout_idx,
    )


def plot_flag_pattern(
    df: pd.DataFrame,
    signal: FlagSignal,
    output_path: str,
) -> None:
    """
    Gera um gráfico simples com:
      - preços de fecho
      - destaque visual do mastro, bandeira e ponto de entrada
      - níveis de SL e TP
    """
    plt.figure(figsize=(10, 5))

    closes = df["Close"]
    dates = df.index

    # Série completa
    plt.plot(dates, closes, label="Close", color="gray", linewidth=1)

    # Mastro
    plt.plot(
        dates[signal.mast_start_idx : signal.mast_end_idx + 1],
        closes.iloc[signal.mast_start_idx : signal.mast_end_idx + 1],
        label="Mastro",
        color="tab:blue",
        linewidth=2,
    )

    # Bandeira
    plt.plot(
        dates[signal.flag_start_idx : signal.flag_end_idx + 1],
        closes.iloc[signal.flag_start_idx : signal.flag_end_idx + 1],
        label="Bandeira",
        color="tab:orange",
        linewidth=2,
    )

    # Ponto de entrada (breakout)
    breakout_price = signal.entry
    plt.scatter(
        dates[signal.breakout_idx],
        breakout_price,
        color="green",
        label="Entry (breakout)",
        zorder=5,
    )

    # Níveis de SL e TP
    plt.axhline(signal.stop_loss, color="red", linestyle="--", linewidth=1, label="Stop Loss")
    plt.axhline(signal.take_profit, color="green", linestyle="--", linewidth=1, label="Take Profit")

    plt.title(f"Padrão de bandeira - {signal.ticker}")
    plt.xlabel("Data")
    plt.ylabel("Preço de fecho")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()
    logging.info("Gráfico do padrão guardado em %s", output_path)


def analyze_universe(
    tickers: List[str],
    params: FlagParams,
    generate_plots: bool = False,
    plots_dir: str = "plots",
    detection_mode: str = "heuristic",
) -> List[FlagSignal]:
    """
    Analisa uma lista de tickers e devolve apenas os que
    apresentam um sinal de compra segundo a estratégia de bandeira.
    """
    signals: List[FlagSignal] = []
    total = len(tickers)
    for i, ticker in enumerate(tickers, start=1):
        logging.info("(%d/%d) A analisar %s...", i, total, ticker)
        try:
            signal = analyze_ticker(ticker, params, detection_mode=detection_mode)
        except Exception as exc:  # noqa: BLE001
            logging.exception("Erro ao analisar %s: %s", ticker, exc)
            continue
        if signal is not None:
            signals.append(signal)
            logging.info(
                "Sinal encontrado em %s | Entry=%.2f SL=%.2f TP=%.2f Range=%.2f Score=%.2f",
                signal.ticker,
                signal.entry,
                signal.stop_loss,
                signal.take_profit,
                signal.range_points,
                signal.score,
            )

            if generate_plots:
                # Reutilizar os dados de preços para o gráfico
                df = get_price_data(
                    ticker=ticker,
                    period="1y",
                    interval="1d",
                    lookback_candles=params.lookback_candles,
                )
                if df.empty:
                    continue

                import os  # local import para evitar overhead se não for usado

                os.makedirs(plots_dir, exist_ok=True)
                safe_ticker = ticker.replace("/", "_").replace(":", "_")
                output_path = os.path.join(plots_dir, f"{safe_ticker}.png")
                plot_flag_pattern(df, signal, output_path)
    return signals


def _format_signals_table(signals: List[FlagSignal]) -> str:
    if not signals:
        return "Nenhum sinal encontrado segundo a estratégia de bandeira."

    def _score_label(score: float) -> str:
        """
        Devolve uma etiqueta textual para a qualidade do padrão,
        baseada no valor do score.
        """
        if score < 5:
            return "baixa"
        if score < 10:
            return "média"
        if score < 20:
            return "intermédia-alta"
        return "alta"

    rows = []
    header = ["Ticker", "Entry", "StopLoss", "TakeProfit", "Range", "Score"]
    rows.append(header)

    for s in signals:
        rows.append(
            [
                s.ticker,
                f"{s.entry:.2f}",
                f"{s.stop_loss:.2f}",
                f"{s.take_profit:.2f}",
                f"{s.range_points:.2f}",
                f"{s.score:.2f} ({_score_label(s.score)})",
            ]
        )

    col_widths = [max(len(str(row[i])) for row in rows) for i in range(len(header))]

    def fmt_row(row: List[str]) -> str:
        return "  ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))

    lines = [
        "Ações com sinal de compra segundo estratégia Flag:",
        "",
        fmt_row(header),
        "-" * (sum(col_widths) + 2 * (len(header) - 1)),
    ]
    for row in rows[1:]:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def save_signals_to_csv(signals: List[FlagSignal], path: str) -> None:
    """
    Exporta os sinais encontrados para um ficheiro CSV.
    """
    if not signals:
        logging.info("Nenhum sinal para exportar para %s", path)
        return

    data: List[Dict[str, object]] = []
    for s in signals:
        data.append(
            {
                "Ticker": s.ticker,
                "Entry": s.entry,
                "StopLoss": s.stop_loss,
                "TakeProfit": s.take_profit,
                "Range": s.range_points,
                "Score": s.score,
                "MastStart": s.mast_start,
                "MastEnd": s.mast_end,
                "FlagStart": s.flag_start,
                "FlagEnd": s.flag_end,
                "BreakoutTime": s.breakout_time,
            }
        )
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    logging.info("Sinais exportados para %s (%d linhas)", path, len(df))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Análise de ações com estratégia de bandeira (flag pattern).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="stock600.csv",
        help="Caminho para o ficheiro CSV com a lista de tickers (coluna 'Ticket' ou 'Ticker'). "
        "Por omissão usa 'stock600.csv' na diretoria atual.",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Lista manual de tickers separada por vírgulas, por exemplo: 'AAPL,MSFT,NVDA'. "
        "Se fornecida, tem prioridade sobre o CSV.",
    )
    parser.add_argument(
        "--pattern-threshold",
        type=float,
        default=0.0,
        help=(
            "Limiar mínimo do score interno do padrão. "
            "Valores mais altos tornam a deteção mais exigente."
        ),
    )
    parser.add_argument(
        "--take-profit-multiplier",
        type=float,
        default=1.0,
        help="Multiplicador do range para o Take Profit (default: 1.0).",
    )
    parser.add_argument(
        "--stop-loss-multiplier",
        type=float,
        default=0.25,
        help="Multiplicador do range para o Stop Loss (default: 0.25).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Se definido, exporta os sinais encontrados para este ficheiro CSV.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Se definido, gera gráficos PNG do padrão detetado para cada sinal.",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="plots",
        help="Diretoria onde guardar os gráficos gerados (default: 'plots').",
    )
    parser.add_argument(
        "--detection-mode",
        type=str,
        choices=("heuristic", "weight_matrix"),
        default="",
        help="Modo de deteção: 'heuristic' ou 'weight_matrix'. Se vazio, será perguntado ao utilizador.",
    )
    parser.add_argument(
        "--min-range-pct",
        type=float,
        default=0.2,
        help="Range mínimo em %% do preço de entrada; sinais abaixo são ignorados (default: 0.2).",
    )
    return parser.parse_args()


def _ask_detection_mode() -> str:
    """Pergunta ao utilizador se quer usar matriz de pesos ou deteção heurística."""
    while True:
        try:
            resp = input(
                "Modo de deteção do padrão bandeira:\n"
                "  (1) Matriz de pesos\n"
                "  (2) Heurística\n"
                "Escolha [1/2] (default: 2): "
            ).strip() or "2"
            if resp == "1":
                return "weight_matrix"
            if resp == "2":
                return "heuristic"
        except (EOFError, KeyboardInterrupt):
            return "heuristic"
        print("  Introduza 1 ou 2.")


def main() -> None:
    args = parse_args()

    if args.detection_mode:
        detection_mode = args.detection_mode
        logging.info("Modo de deteção: %s", detection_mode)
    else:
        detection_mode = _ask_detection_mode()
        logging.info("Modo de deteção escolhido: %s", detection_mode)

    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
        logging.info("A usar lista manual de %d tickers.", len(tickers))
    else:
        tickers = load_tickers_from_csv(args.csv)

    params = FlagParams(
        pattern_threshold=args.pattern_threshold,
        take_profit_multiplier=args.take_profit_multiplier,
        stop_loss_multiplier=args.stop_loss_multiplier,
        min_range_pct=args.min_range_pct,
    )

    signals = analyze_universe(
        tickers,
        params,
        generate_plots=args.plots,
        plots_dir=args.plots_dir,
        detection_mode=detection_mode,
    )

    print()
    print(_format_signals_table(signals))
    print()
    print(f"Total de ações analisadas: {len(tickers)}")
    print(f"Total de sinais encontrados: {len(signals)}")

    if args.output_csv:
        save_signals_to_csv(signals, args.output_csv)


if __name__ == "__main__":
    main()

