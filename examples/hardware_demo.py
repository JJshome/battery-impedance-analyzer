#!/usr/bin/env python3
"""
Demo script for interacting with the Ucaretron Impedance SoC hardware.

This script demonstrates:
1. Connecting to the hardware (or simulator)
2. Configuring the hardware
3. Running diagnostics and calibration
4. Performing impedance measurements
5. Analyzing the results with the ImpedanceAnalyzer

Usage:
    python hardware_demo.py

Author: Ucaretron Inc.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our package
from battery_analyzer import ImpedanceAnalyzer
from battery_analyzer.hardware_interface import create_hardware_interface

def main():
    print("Ucaretron Impedance SoC Hardware Demo")
    print("=====================================")
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Create a hardware interface (using simulator by default)
    print("\nInitializing hardware interface...")
    hardware = create_hardware_interface(
        use_simulator=True,
        device_id="UcaretronEVA-001",
        # Simulated battery parameters:
        r_s=0.028,      # Series resistance (Ohms)
        r_ct=0.017,     # Charge transfer resistance (Ohms)
        c_dl=0.085,     # Double layer capacitance (Farads)
        r_w=0.022,      # Warburg resistance (Ohms)
        t_w=1.8,        # Warburg time constant (seconds)
        soh=90.0,       # State of Health (%)
        temperature=30.0  # Temperature (°C)
    )
    
    # Connect to the hardware
    print("Connecting to hardware...")
    if not hardware.connect():
        print("Failed to connect to hardware. Exiting.")
        return
    
    # Get hardware status
    status = hardware.get_status()
    print("\nHardware Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Run diagnostics
    print("\nRunning hardware diagnostics...")
    diag_results = hardware.run_diagnostics()
    if diag_results['success']:
        print("Diagnostics passed successfully!")
    else:
        print(f"Diagnostics failed: {diag_results.get('message', 'Unknown error')}")
        hardware.disconnect()
        return
    
    # Calibrate the hardware
    print("\nCalibrating hardware...")
    if not hardware.calibrate():
        print("Calibration failed. Exiting.")
        hardware.disconnect()
        return
    print("Calibration completed successfully!")
    
    # Configure the hardware for a frequency sweep measurement
    print("\nConfiguring hardware for impedance measurement...")
    hardware.configure(
        measurement_mode='FREQUENCY_SWEEP',
        start_frequency=0.1,     # Hz (Lower limit for faster demo)
        end_frequency=10000,     # Hz
        points_per_decade=10,    # More points for better resolution
        amplitude=0.2,           # A
        temperature_compensation=True
    )
    
    # Perform impedance measurement
    print("\nPerforming impedance measurement...")
    measurement_data = hardware.measure_impedance()
    if measurement_data is None:
        print("Measurement failed. Exiting.")
        hardware.disconnect()
        return
    
    print(f"Measurement completed: {len(measurement_data['frequency'])} frequency points.")
    
    # Create an analyzer instance and load the measurement data
    analyzer = ImpedanceAnalyzer()
    analyzer.load_data_from_array(
        measurement_data['frequency'],
        measurement_data['real'],
        measurement_data['imag']
    )
    
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
            print(f"  {anomaly.get('description', '')}")
    
    # Generate plots
    print("\nGenerating plots...")
    nyquist_path = os.path.join('output', 'hardware_nyquist_plot.png')
    analyzer.plot_nyquist(save_path=nyquist_path)
    print(f"Nyquist plot saved to {nyquist_path}")
    
    bode_path = os.path.join('output', 'hardware_bode_plot.png')
    analyzer.plot_bode(save_path=bode_path)
    print(f"Bode plot saved to {bode_path}")
    
    # Generate a report
    report_path = os.path.join('output', 'hardware_measurement_report.md')
    analyzer.generate_report(results, report_path)
    print(f"\nDetailed report saved to {report_path}")
    
    # Demonstrate multiple SOH measurements with different simulated battery states
    print("\nDemonstrating SOH tracking with different battery states...")
    demonstrate_soh_tracking(hardware, analyzer)
    
    # Clean up
    print("\nDisconnecting from hardware...")
    hardware.disconnect()
    print("Demo completed!")

def demonstrate_soh_tracking(hardware, analyzer):
    """
    Demonstrate SOH tracking with different simulated battery states.
    
    Parameters:
    -----------
    hardware : Hardware interface instance
    analyzer : ImpedanceAnalyzer instance
    """
    # Define SOH values to simulate
    soh_values = [95, 90, 85, 80, 75, 70]
    measured_soh = []
    
    # Set up plot
    plt.figure(figsize=(10, 6))
    
    # Measure impedance and calculate SOH for each simulated state
    for i, soh in enumerate(soh_values):
        print(f"\nSimulating battery with SOH = {soh}%")
        
        # Update simulator battery parameters for degraded battery
        if hasattr(hardware, 'set_battery_parameters'):  # Only works with simulator
            # Simulate degradation in battery parameters
            degradation_factor = (100 - soh) / 30  # Scale factor
            hardware.set_battery_parameters(
                soh=soh,
                r_s=0.028 * (1 + 0.5 * degradation_factor),
                r_ct=0.017 * (1 + 1.5 * degradation_factor),
                c_dl=0.085 * (1 - 0.3 * degradation_factor),
                r_w=0.022 * (1 + 1.0 * degradation_factor)
            )
        
        # Perform measurement
        measurement_data = hardware.measure_impedance()
        if measurement_data is None:
            print("  Measurement failed.")
            continue
        
        # Load data into analyzer
        analyzer.load_data_from_array(
            measurement_data['frequency'],
            measurement_data['real'],
            measurement_data['imag']
        )
        
        # Analyze
        results = analyzer.analyze()
        estimated_soh = results.get('soh', 0)
        measured_soh.append(estimated_soh)
        
        print(f"  True SOH: {soh}%, Estimated SOH: {estimated_soh:.1f}%")
        
        # Plot the Nyquist diagram for this measurement
        plt.plot(measurement_data['real'], -measurement_data['imag'], 'o-', 
                 linewidth=1, markersize=4, label=f"SOH {soh}%")
    
    # Finalize and save the comparative Nyquist plot
    plt.xlabel('Z\' (Ω)', fontsize=12)
    plt.ylabel('-Z\'\' (Ω)', fontsize=12)
    plt.title('Nyquist Plots for Different Battery States', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.axis('equal')
    
    comparison_path = os.path.join('output', 'soh_comparison_nyquist.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison Nyquist plot saved to {comparison_path}")
    
    # Plot true vs. estimated SOH
    plt.figure(figsize=(8, 6))
    plt.plot(soh_values, soh_values, 'k--', label='Perfect estimation')
    plt.plot(soh_values, measured_soh, 'ro-', linewidth=2, label='Estimated SOH')
    
    plt.xlabel('True SOH (%)', fontsize=12)
    plt.ylabel('Estimated SOH (%)', fontsize=12)
    plt.title('SOH Estimation Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    soh_accuracy_path = os.path.join('output', 'soh_estimation_accuracy.png')
    plt.savefig(soh_accuracy_path, dpi=300, bbox_inches='tight')
    print(f"SOH estimation accuracy plot saved to {soh_accuracy_path}")

if __name__ == "__main__":
    main()
