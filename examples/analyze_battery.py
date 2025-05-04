#!/usr/bin/env python3
"""
Example script demonstrating the use of the Battery Impedance Analyzer.

This script:
1. Loads sample impedance data
2. Analyzes the battery health
3. Generates plots and a report
4. Demonstrates how to use the analyzer with real-time data (simulated)

Usage:
    python analyze_battery.py

Author: Ucaretron Inc.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our package
from battery_analyzer import ImpedanceAnalyzer

def main():
    print("Battery Impedance Analysis Example")
    print("==================================")
    
    # Create directories for output if they don't exist
    os.makedirs('output', exist_ok=True)
    
    # Create an instance of the impedance analyzer
    analyzer = ImpedanceAnalyzer()
    
    # Determine the path to the sample data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'sample_impedance_data.csv')
    
    print(f"\nLoading impedance data from {data_path}")
    analyzer.load_data(data_path)
    
    # Analyze the data
    print("\nAnalyzing battery health...")
    results = analyzer.analyze()
    
    # Print summary of results
    print("\nAnalysis Results:")
    print(f"State of Health (SOH): {results.get('soh', 'N/A'):.1f}%")
    
    if 'rul' in results and results['rul']:
        print(f"Estimated remaining cycles: {results['rul'].get('remaining_cycles', 'N/A'):.0f}")
        print(f"Estimated remaining months: {results['rul'].get('remaining_months', 'N/A'):.1f}")
    
    if 'anomalies' in results and results['anomalies']:
        print("\nDetected anomalies:")
        for anomaly in results['anomalies']:
            print(f"- {anomaly.get('type', 'Unknown')}: {anomaly.get('severity', 'unknown')} severity")
    
    # Generate and save Nyquist plot
    print("\nGenerating plots...")
    nyquist_path = os.path.join('output', 'nyquist_plot.png')
    analyzer.plot_nyquist(save_path=nyquist_path)
    print(f"Nyquist plot saved to {nyquist_path}")
    
    # Generate and save Bode plot
    bode_path = os.path.join('output', 'bode_plot.png')
    analyzer.plot_bode(save_path=bode_path)
    print(f"Bode plot saved to {bode_path}")
    
    # Generate a report
    report_path = os.path.join('output', 'battery_health_report.md')
    analyzer.generate_report(results, report_path)
    print(f"\nDetailed report saved to {report_path}")
    
    # Demonstrate real-time monitoring (simulated)
    print("\nSimulating real-time monitoring...")
    simulate_real_time_monitoring(analyzer)
    
    print("\nAnalysis complete!")

def simulate_real_time_monitoring(analyzer):
    """
    Simulate real-time monitoring of a battery with gradually degrading impedance.
    This is just a demonstration of how the analyzer could be used with real-time data.
    
    Parameters:
    -----------
    analyzer : ImpedanceAnalyzer
        The analyzer instance to use
    """
    # Load the original data as a baseline
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'sample_impedance_data.csv')
    
    # In a real application, this would come from the hardware
    import pandas as pd
    original_data = pd.read_csv(data_path)
    
    # Simulate 5 measurements over time with increasing degradation
    degradation_factors = [1.0, 1.05, 1.1, 1.15, 1.2]
    soh_values = []
    timestamps = []
    
    for i, factor in enumerate(degradation_factors):
        # Simulate degradation by increasing the real part of impedance
        modified_data = original_data.copy()
        modified_data['real'] = original_data['real'] * factor
        modified_data['|Z|'] = np.sqrt(modified_data['real']**2 + modified_data['imag']**2)
        
        # Update the analyzer with the new data
        analyzer.load_data_from_array(
            modified_data['frequency'].values,
            modified_data['real'].values,
            modified_data['imag'].values
        )
        
        # Analyze and store results
        results = analyzer.analyze()
        soh = results.get('soh', 100.0)
        soh_values.append(soh)
        
        # Simulate timestamps (one month apart)
        timestamp = datetime.now().replace(month=(datetime.now().month + i) % 12 or 12)
        timestamps.append(timestamp)
        
        # Print update
        print(f"Measurement {i+1}: SOH = {soh:.1f}%")
    
    # Plot SOH trend
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(soh_values)), soh_values, 'o-', linewidth=2)
    plt.xlabel('Measurement Number')
    plt.ylabel('State of Health (%)')
    plt.title('Battery Health Degradation Trend')
    plt.grid(True, alpha=0.3)
    plt.ylim(min(soh_values) - 5, 105)
    
    # Add simple trend line
    z = np.polyfit(range(len(soh_values)), soh_values, 1)
    p = np.poly1d(z)
    plt.plot(range(len(soh_values)), p(range(len(soh_values))), "r--", alpha=0.8)
    
    # Project to EOL (80% SOH)
    if soh_values[-1] > 80:
        # Calculate when SOH will reach 80%
        eol_measurement = (80 - z[1]) / z[0]
        months_to_eol = eol_measurement - len(soh_values) + 1
        
        # Add EOL point to plot
        plt.plot(eol_measurement, 80, 'rx', markersize=10)
        plt.axhline(y=80, color='r', linestyle=':', alpha=0.5)
        plt.axvline(x=eol_measurement, color='r', linestyle=':', alpha=0.5)
        
        plt.text(eol_measurement + 0.1, 80 - 2, f"EOL: {months_to_eol:.1f} months", 
                 bbox=dict(facecolor='white', alpha=0.8))
    
    trend_path = os.path.join('output', 'soh_trend.png')
    plt.savefig(trend_path, dpi=300, bbox_inches='tight')
    print(f"SOH trend saved to {trend_path}")

if __name__ == "__main__":
    main()
