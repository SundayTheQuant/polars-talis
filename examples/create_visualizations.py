"""
Example script to create visualizations for polars-talis technical indicators.
This script generates sample data and creates professional charts showing the indicators in action.
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
from pathlib import Path

# Import our package
import sys
sys.path.append('..')
from polars_talis import TechnicalAnalyzer, SMA, EMA, RSI, MACD, BollingerBands

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def generate_sample_data(days=365, start_price=100):
    """Generate realistic financial time series data"""
    np.random.seed(42)
    
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Generate more realistic price movements
    returns = np.random.normal(0.0005, 0.02, days)  # Small positive drift with volatility
    prices = [start_price]
    
    for i in range(1, days):
        # Add some trending and mean reversion
        trend = 0.0001 * np.sin(i * 2 * np.pi / 90)  # Quarterly cycles
        mean_reversion = -0.001 * (prices[-1] - 100) / 100  # Mean revert to 100
        price_change = returns[i] + trend + mean_reversion
        new_price = prices[-1] * (1 + price_change)
        prices.append(max(new_price, 10))  # Prevent negative prices
    
    volumes = np.random.randint(50000, 200000, days).astype(int)
    
    return pl.DataFrame({
        "date": dates,
        "close": prices,
        "volume": volumes
    }, strict=False)

def create_price_and_trend_chart(df, save_path="../images/price_trends.png"):
    """Create a chart showing price with SMA and EMA indicators"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Convert dates for matplotlib
    dates = [d.date() for d in df['date'].to_list()]
    
    # Plot price and moving averages
    ax.plot(dates, df['close'].to_list(), label='Close Price', linewidth=2, color='#2E86AB')
    ax.plot(dates, df['SMA_20'].to_list(), label='SMA(20)', linewidth=2, color='#A23B72', alpha=0.8)
    ax.plot(dates, df['SMA_50'].to_list(), label='SMA(50)', linewidth=2, color='#F18F01', alpha=0.8)
    ax.plot(dates, df['EMA_12'].to_list(), label='EMA(12)', linewidth=2, color='#C73E1D', alpha=0.8)
    
    ax.set_title('Price Action with Trend Indicators', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Price and trend chart saved to {save_path}")

def create_bollinger_bands_chart(df, save_path="../images/bollinger_bands.png"):
    """Create a chart showing Bollinger Bands"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    dates = [d.date() for d in df['date'].to_list()]
    
    # Plot Bollinger Bands - filter out null values
    bb_lower = [x if x is not None else np.nan for x in df['BB_lower'].to_list()]
    bb_upper = [x if x is not None else np.nan for x in df['BB_upper'].to_list()]
    ax.fill_between(dates, bb_lower, bb_upper, 
                   alpha=0.2, color='#A23B72', label='Bollinger Bands')
    ax.plot(dates, df['close'].to_list(), label='Close Price', linewidth=2, color='#2E86AB')
    ax.plot(dates, df['BB_middle'].to_list(), label='BB Middle (SMA 20)', 
           linewidth=2, color='#F18F01', linestyle='--', alpha=0.8)
    ax.plot(dates, bb_upper, label='BB Upper', 
           linewidth=1, color='#A23B72', alpha=0.8)
    ax.plot(dates, bb_lower, label='BB Lower', 
           linewidth=1, color='#A23B72', alpha=0.8)
    
    ax.set_title('Bollinger Bands - Volatility Analysis', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Bollinger Bands chart saved to {save_path}")

def create_momentum_indicators_chart(df, save_path="../images/momentum_indicators.png"):
    """Create a chart showing RSI and MACD"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), 
                                       gridspec_kw={'height_ratios': [2, 1, 1]})
    
    dates = [d.date() for d in df['date'].to_list()]
    
    # Price chart
    ax1.plot(dates, df['close'].to_list(), label='Close Price', linewidth=2, color='#2E86AB')
    ax1.set_title('Price Action with Momentum Indicators', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # RSI chart
    ax2.plot(dates, df['RSI_14'].to_list(), label='RSI(14)', linewidth=2, color='#C73E1D')
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
    ax2.fill_between(dates, 30, 70, alpha=0.1, color='gray')
    ax2.set_ylabel('RSI', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # MACD chart
    ax3.plot(dates, df['MACD_line'].to_list(), label='MACD Line', linewidth=2, color='#A23B72')
    ax3.plot(dates, df['MACD_signal'].to_list(), label='Signal Line', linewidth=2, color='#F18F01')
    
    # MACD histogram
    histogram = (df['MACD_histogram'].to_list())
    colors = ['green' if x >= 0 else 'red' for x in histogram]
    ax3.bar(dates, histogram, alpha=0.3, color=colors, label='MACD Histogram')
    
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('MACD', fontsize=12)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis for all subplots
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Momentum indicators chart saved to {save_path}")

def create_performance_comparison_chart(save_path="../images/performance_comparison.png"):
    """Create a chart comparing performance with different dataset sizes and indicator counts"""
    import time
    
    print("📊 Running comprehensive performance benchmarks...")
    
    # Test with much larger dataset sizes to find the real crossover point
    dataset_sizes = [10000, 50000, 100000, 250000, 500000, 1000000]
    parallel_speedup = []
    
    print("Testing dataset size impact (with 5 indicators):")
    for size in dataset_sizes:
        df = generate_sample_data(size)
        
        # Use multiple indicators for meaningful parallel work
        indicators = [
            SMA(20), SMA(50), 
            EMA(12), EMA(26),
            RSI(14),
            MACD(),
            BollingerBands(20, 2.0)
        ]
        
        # Sequential timing
        analyzer_seq = TechnicalAnalyzer(max_workers=1)
        analyzer_seq.add_indicators(indicators)
        
        start_time = time.time()
        result_seq = analyzer_seq.calculate(df, parallel=False)
        sequential_time = time.time() - start_time
        
        # Parallel timing
        analyzer_par = TechnicalAnalyzer(max_workers=4)
        analyzer_par.add_indicators(indicators)
        
        start_time = time.time()
        result_par = analyzer_par.calculate(df, parallel=True)
        parallel_time = time.time() - start_time
        
        speedup = sequential_time / parallel_time if parallel_time > 0 else 1
        parallel_speedup.append(speedup)
        
        print(f"  {size:,} rows: Sequential={sequential_time:.3f}s, Parallel={parallel_time:.3f}s, Speedup={speedup:.2f}x")
    
    # Create the performance chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Chart 1: Speedup vs Dataset Size
    ax1.plot(dataset_sizes, parallel_speedup, 'o-', linewidth=3, markersize=8, color='#2E86AB')
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup (1.0x)')
    ax1.set_xlabel('Dataset Size (rows)', fontsize=12)
    ax1.set_ylabel('Parallel Speedup (x times faster)', fontsize=12)
    ax1.set_title('Parallel Processing Speedup vs Dataset Size', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Format x-axis labels
    ax1.set_xticks(dataset_sizes)
    ax1.set_xticklabels([f'{x//1000}K' if x < 1000000 else f'{x//1000000}M' for x in dataset_sizes])
    
    # Add speedup annotations
    for i, (size, speedup) in enumerate(zip(dataset_sizes, parallel_speedup)):
        ax1.annotate(f'{speedup:.2f}x', 
                    xy=(size, speedup), 
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontweight='bold')
    
    # Chart 2: Processing time comparison for largest dataset
    large_df = generate_sample_data(500000)  # Use 500K rows for more realistic test
    indicators_count = [3, 6, 9, 12, 15]  # More indicators for better parallelization
    seq_times = []
    par_times = []
    
    print("\nTesting indicator count impact (with 500K rows):")
    for count in indicators_count:
        indicators = [
            SMA(20), SMA(50), SMA(100), EMA(12), EMA(26), EMA(50),
            RSI(14), RSI(7), RSI(21), MACD(), BollingerBands(20, 2.0), 
            BollingerBands(10, 1.5), SMA(200), EMA(100), RSI(28)
        ][:count]
        
        # Sequential
        analyzer_seq = TechnicalAnalyzer(max_workers=1)
        analyzer_seq.add_indicators(indicators)
        start_time = time.time()
        analyzer_seq.calculate(large_df, parallel=False)
        seq_time = time.time() - start_time
        seq_times.append(seq_time)
        
        # Parallel
        analyzer_par = TechnicalAnalyzer(max_workers=4)
        analyzer_par.add_indicators(indicators)
        start_time = time.time()
        analyzer_par.calculate(large_df, parallel=True)
        par_time = time.time() - start_time
        par_times.append(par_time)
        
        print(f"  {count} indicators: Sequential={seq_time:.3f}s, Parallel={par_time:.3f}s")
    
    x = np.arange(len(indicators_count))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, seq_times, width, label='Sequential', alpha=0.8, color='#C73E1D')
    bars2 = ax2.bar(x + width/2, par_times, width, label='Parallel', alpha=0.8, color='#2E86AB')
    
    ax2.set_xlabel('Number of Indicators', fontsize=12)
    ax2.set_ylabel('Processing Time (seconds)', fontsize=12)
    ax2.set_title('Processing Time: 500K Rows Dataset', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(indicators_count)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Comprehensive performance comparison chart saved to {save_path}")

def main():
    """Main function to generate all visualizations"""
    print("🎨 Creating visualizations for polars-talis...")
    
    # Generate sample data
    print("📊 Generating sample financial data...")
    df = generate_sample_data(365)
    
    # Create analyzer and calculate indicators
    print("🔧 Calculating technical indicators...")
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
    
    result = analyzer.calculate(df, parallel=True)
    
    # Create all charts
    print("📈 Creating charts...")
    create_price_and_trend_chart(result)
    create_bollinger_bands_chart(result)
    create_momentum_indicators_chart(result)
    
    print("\n🎉 All visualizations created successfully!")
    print("📁 Charts saved in the 'images/' directory:")
    print("   • images/price_trends.png")
    print("   • images/bollinger_bands.png") 
    print("   • images/momentum_indicators.png")

if __name__ == "__main__":
    main()