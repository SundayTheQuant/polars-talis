import polars as pl
from typing import Optional, List
from ..core.base import BaseIndicator, IndicatorConfig, IndicatorType


class MACD(BaseIndicator):
    """Moving Average Convergence Divergence"""
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, column: str = "close", name: Optional[str] = None):
        config = IndicatorConfig(
            name=name or f"MACD",
            type=IndicatorType.MOMENTUM,
            params={
                "fast": fast,
                "slow": slow,
                "signal": signal,
                "column": column
            },
            output_columns=[
                f"{name or 'MACD'}_line",
                f"{name or 'MACD'}_signal",
                f"{name or 'MACD'}_histogram"
            ]
        )
        super().__init__(config)

    def _validate_config(self) -> None:
        params = self.config.params
        if params["fast"] >= params["slow"]:
            raise ValueError("Fast period must be less than slow period.")

    def _calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        params = self.config.params
        column = params["column"]

        alpha_fast = 2 / (params["fast"] + 1)
        alpha_slow = 2 / (params["slow"] + 1)
        alpha_signal = 2 / (params["signal"] + 1)

        result = df.with_columns([
            pl.col(column).ewm_mean(alpha=alpha_fast).alias("_ema_fast"),
            pl.col(column).ewm_mean(alpha=alpha_slow).alias("_ema_slow")
        ]).with_columns([
            (pl.col("_ema_fast") - pl.col("_ema_slow")).alias(f"{self.config.output_columns[0]}"),
        ]).with_columns([
            pl.col(f"{self.config.output_columns[0]}").ewm_mean(alpha=alpha_signal).alias(f"{self.config.output_columns[1]}"),
        ]).with_columns([
            (pl.col(self.config.output_columns[0]) - pl.col(self.config.output_columns[1])).alias(f"{self.config.output_columns[2]}"),
        ]).drop("_ema_fast", "_ema_slow")

        return result

    def get_required_columns(self) -> List[str]:
        return [self.config.params["column"]]


class RSI(BaseIndicator):
    """Relative Strength Index"""

    def __init__(self, period: int = 14, column: str = "close", name: Optional[str] = None):
        config = IndicatorConfig(
            name=name or f"RSI_{period}",
            type=IndicatorType.MOMENTUM,
            params={"period": period, "column": column},
            output_columns=[name or f"RSI_{period}"]
        )
        super().__init__(config)

    def _validate_config(self) -> None:
        if self.config.params["period"] <= 0:
            raise ValueError("Period must be positive")

    def _calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        column = self.config.params["column"]
        period = self.config.params["period"]
        output_name = self.config.output_columns[0]

        return df.with_columns([
            pl.col(column).diff().alias("_price_change")
        ]).with_columns([
            pl.when(pl.col("_price_change") > 0)
            .then(pl.col("_price_change"))
            .otherwise(0).alias("_gains"),
            pl.when(pl.col("_price_change") < 0)
            .then(-pl.col("_price_change"))
            .otherwise(0).alias("_losses")
        ]).with_columns([
            pl.col("_gains").rolling_mean(window_size=period).alias("_avg_gains"),
            pl.col("_losses").rolling_mean(window_size=period).alias("_avg_losses")
        ]).with_columns([
            (100 - (100 / (1 + pl.col("_avg_gains") / pl.col("_avg_losses"))))
            .alias(output_name)
        ]).drop(["_price_change", "_gains", "_losses", "_avg_gains", "_avg_losses"])

    def get_required_columns(self) -> List[str]:
        return [self.config.params["column"]]