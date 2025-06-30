import polars as pl
from typing import Optional, List
from core.base import BaseIndicator, IndicatorConfig, IndicatorType

class SMA(BaseIndicator):
    """Simple Moving Average"""
    def __init__(self, period: int, column: str = "close", name: Optional[str] = None):
        config = IndicatorConfig(
            name=name or f"SMA_{period}",
            type=IndicatorType.TREND,
            params={"period": period, "column": column},
            output_columns=[name or f"SMA_{period}"]
        )
        super().__init__(config)

    def _validate_config(self) -> None:
        if self.config.params["period"] <= 0:
            raise ValueError("Period must be a positive integer.")

    def _calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        column = self.config.params["column"]
        period = self.config.params["period"]
        output_name = self.config.output_columns[0]
        return df.with_columns([
            pl.col(column).rolling_mean(window_size=period).alias(output_name)
        ])

    def get_required_columns(self) -> List[str]:
        return [self.config.params["column"]]


class EMA(BaseIndicator):
    """Exponential Moving Average"""
    def __init__(self, period: int, column: str = "close", name: Optional[str] = None):
        config = IndicatorConfig(
            name=name or f"EMA_{period}",
            type=IndicatorType.TREND,
            params={"period": period, "column": column},
            output_columns=[name or f"EMA_{period}"]
        )
        super().__init__(config)

    def _validate_config(self) -> None:
        if self.config.params["period"] <= 0:
            raise ValueError("Period must be a positive integer.")

    def _calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        column = self.config.params["column"]
        period = self.config.params["period"]
        output_name = self.config.output_columns[0]
        alpha = 2 / (period + 1)
        return df.with_columns([
            pl.col(column).ewm_mean(alpha=alpha).alias(output_name)
        ])

    def get_required_columns(self) -> List[str]:
        return [self.config.params["column"]]


