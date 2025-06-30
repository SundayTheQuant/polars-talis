from core.analyzer import TechnicalAnalyzer
from indicators.trend import SMA, EMA
from indicators.momentum import MACD, RSI
from indicators.volatility import BollingerBands

if __name__ == "__main__":
    import polars as pl
    import numpy as np
    from datetime import datetime, timedelta

    # Generar datos de ejemplo
    np.random.seed(42)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    num_days = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    prices = 100 + np.cumsum(np.random.randn(num_days) * 0.5)

    df = pl.DataFrame({
        "date": dates,
        "close": prices,
        "volume": np.random.randint(1000, 10000, len(dates))
    })

    # Crear analizador con múltiples indicadores
    analyzer = TechnicalAnalyzer(max_workers=4)
    analyzer.add_indicators([
        SMA(20),
        SMA(50),
        EMA(12),
        EMA(26),
        MACD(),
        RSI(14),
        BollingerBands(20, 2.0)
    ])

    # Calcular todos los indicadores
    result = analyzer.calculate(df, parallel=True)

    print("Columnas resultantes:")
    print(result.columns)
    print("\nResumen del analizador:")
    print(analyzer.get_summary())
    print(f"\nDatos procesados: {len(result)} filas")
    print(result)
    print(result.to_numpy().T)