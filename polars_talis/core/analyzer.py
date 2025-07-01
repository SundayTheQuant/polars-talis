from typing import List, Dict, Any, Optional
import polars as pl
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from .base import BaseIndicator

class TechnicalAnalyzer:
    """Analizador principal que ejecuta múltiples indicadores"""

    def __init__(self, max_workers: Optional[int] = None):
        self.indicators: List[BaseIndicator] = []
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)

    def add_indicator(self, indicator: BaseIndicator) -> 'TechnicalAnalyzer':
        """Agrega un indicador al analizador"""
        self.indicators.append(indicator)
        return self

    def add_indicators(self, indicators: List[BaseIndicator]) -> 'TechnicalAnalyzer':
        """Agrega múltiples indicadores"""
        self.indicators.extend(indicators)
        return self

    def validate_data(self, df: pl.DataFrame) -> None:
        """Valida que el DataFrame tenga las columnas requeridas"""
        required_columns = set()
        for indicator in self.indicators:
            required_columns.update(indicator.get_required_columns())

        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def calculate_single(self, df: pl.DataFrame, indicator: BaseIndicator) -> pl.DataFrame:
        """Calcula un solo indicador"""
        try:
            return indicator._calculate(df)
        except Exception as e:
            self.logger.error(f"Error calculating {indicator}: {e}")
            raise

    def calculate_parallel(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calcula todos los indicadores en paralelo"""
        if not self.indicators:
            return df

        self.validate_data(df)

        # Agrupar indicadores por dependencias para optimizar ejecución
        independent_indicators = []
        dependent_indicators = []

        for indicator in self.indicators:
            # Verificar si el indicador depende de outputs de otros indicadores
            required_cols = set(indicator.get_required_columns())
            output_cols = set()
            for other in self.indicators:
                if other != indicator:
                    output_cols.update(other.config.output_columns)

            if required_cols.intersection(output_cols):
                dependent_indicators.append(indicator)
            else:
                independent_indicators.append(indicator)

        result_df = df

        # Ejecutar indicadores independientes en paralelo
        if independent_indicators:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.calculate_single, result_df, indicator): indicator
                    for indicator in independent_indicators
                }

                for future in as_completed(futures):
                    indicator = futures[future]
                    try:
                        indicator_result = future.result()
                        # Combinar resultados manteniendo solo las nuevas columnas
                        new_columns = [col for col in indicator_result.columns
                                       if col not in result_df.columns]
                        if new_columns:
                            result_df = result_df.with_columns([
                                indicator_result.select(new_columns).to_series(i)
                                for i, _ in enumerate(new_columns)
                            ])
                    except Exception as e:
                        self.logger.error(f"Failed to calculate {indicator}: {e}")
                        raise

        # Ejecutar indicadores dependientes secuencialmente
        for indicator in dependent_indicators:
            result_df = self.calculate_single(result_df, indicator)

        return result_df

    def calculate(self, df: pl.DataFrame, parallel: bool = True) -> pl.DataFrame:
        """Calcula todos los indicadores"""
        if parallel:
            return self.calculate_parallel(df)
        else:
            result_df = df
            for indicator in self.indicators:
                result_df = self.calculate_single(result_df, indicator)
            return result_df

    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumen de indicadores configurados"""
        summary = {
            "total_indicators": len(self.indicators),
            "by_type": {},
            "indicators": []
        }

        for indicator in self.indicators:
            indicator_type = indicator.config.type.value
            summary["by_type"][indicator_type] = summary["by_type"].get(indicator_type, 0) + 1
            summary["indicators"].append({
                "name": indicator.config.name,
                "type": indicator_type,
                "output_columns": indicator.config.output_columns
            })

        return summary