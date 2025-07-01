import polars as pl
from typing import Optional, List
from ..core.base import BaseIndicator, IndicatorConfig, IndicatorType


class BollingerBands(BaseIndicator):
    """Bollinger Bands"""

    def __init__(self, period: int = 20, std_dev: float = 2.0,
                 column: str = "close", name: Optional[str] = None):
        config = IndicatorConfig(
            name=name or f"BB_{period}",
            type=IndicatorType.VOLATILITY,
            params={"period": period, "std_dev": std_dev, "column": column},
            output_columns=[
                f"{name or 'BB'}_upper",
                f"{name or 'BB'}_middle",
                f"{name or 'BB'}_lower"
            ]
        )
        super().__init__(config)

    def _validate_config(self) -> None:
        params = self.config.params
        if params["period"] <= 0:
            raise ValueError("Period must be positive")
        if params["std_dev"] <= 0:
            raise ValueError("Standard deviation must be positive")

    def _calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        params = self.config.params
        column = params["column"]
        period = params["period"]
        std_dev = params["std_dev"]

        return df.with_columns([
            pl.col(column).rolling_mean(window_size=period).alias("_sma"),
            pl.col(column).rolling_std(window_size=period).alias("_std")
        ]).with_columns([
            pl.col("_sma").alias(self.config.output_columns[1]),  # middle
            (pl.col("_sma") + pl.col("_std") * std_dev).alias(self.config.output_columns[0]),  # upper
            (pl.col("_sma") - pl.col("_std") * std_dev).alias(self.config.output_columns[2])  # lower
        ]).drop(["_sma", "_std"])

    def get_required_columns(self) -> List[str]:
        return [self.config.params["column"]]