# pandas_ta/core/base.py
import polars as pl
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from enum import Enum


class IndicatorType(Enum):
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"


@dataclass
class IndicatorConfig:
    """ Base configuration for indicators"""
    name: str
    type: IndicatorType
    params: Dict[str, Any] = field(default_factory=dict)
    output_columns: List[str] = field(default_factory=list)

class BaseIndicator(ABC):
    """Base Indicator class for all technical indicators."""
    def __init__(self, config: IndicatorConfig):
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the configuration of the indicator."""
        pass

    @abstractmethod
    def _calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate the indicator."""
        pass

    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """Get the required columns for the indicator."""
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.config.name})"