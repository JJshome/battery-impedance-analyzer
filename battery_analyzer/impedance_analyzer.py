import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from typing import Dict, List, Tuple, Union, Optional
import logging

class ImpedanceAnalyzer:
    """
    Main class for battery impedance analysis using Electrochemical Impedance Spectroscopy (EIS).
    
    This class implements methods for:
    - Loading and preprocessing impedance data
    - Fitting equivalent circuit models to impedance spectra
    - Analyzing impedance patterns
    - Estimating battery State of Health (SOH)
    - Predicting Remaining Useful Life (RUL)
    - Detecting anomalies and potential failure modes
    - Cell grouping for battery pack optimization
    
    References:
    - Electrochemical impedance measurement and analysis techniques
    - Multi-dimensional correlation method for battery performance characteristics
    - Pattern matching techniques for cell grouping
    - Real-time vulcanization control and composition material optimization
    """
    
    def __init__(self, freq_range: Tuple[float, float] = (0.01, 10000)):
        """
        Initialize the Impedance Analyzer.
        
        Parameters:
        -----------
        freq_range : Tuple[float, float]
            Frequency range for impedance measurements in Hz.
            Default is 0.01 Hz to 10 kHz (10 mHz to 10 kHz).
        """
        self.freq_range = freq_range
        self.data = None
        self.equivalent_circuit_params = None
        self.soh_estimate = None
        self.rul_prediction = None
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, file_path: str) -> None:
        """
        Load impedance data from a CSV file.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing impedance data.
            Expected columns: frequency, real, imag, |Z|, phase
        """
        try:
            self.data = pd.read_csv(file_path)
            self.logger.info(f"Successfully loaded data from {file_path}")
            self._preprocess_data()
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def load_data_from_array(self, 
                           frequencies: np.ndarray, 
                           real_part: np.ndarray, 
                           imag_part: np.ndarray) -> None:
        """
        Load impedance data from arrays.
        
        Parameters:
        -----------
        frequencies : np.ndarray
            Array of frequencies in Hz
        real_part : np.ndarray
            Array of real parts of impedance in Ohms
        imag_part : np.ndarray
            Array of imaginary parts of impedance in Ohms
        """
        try:
            self.data = pd.DataFrame({
                'frequency': frequencies,
                'real': real_part,
                'imag': imag_part,
                '|Z|': np.sqrt(real_part**2 + imag_part**2),
                'phase': np.arctan2(imag_part, real_part) * 180 / np.pi
            })
            self.logger.info("Successfully loaded data from arrays")
            self._preprocess_data()
        except Exception as e:
            self.logger.error(f"Error loading data from arrays: {str(e)}")
            raise
    
    def _preprocess_data(self) -> None:
        """
        Preprocess the loaded impedance data:
        - Sort by frequency
        - Remove noise
        - Apply temperature compensation if temperature data is available
        """
        if self.data is None:
            self.logger.error("No data to preprocess")
            return
            
        # Sort by frequency
        self.data = self.data.sort_values('frequency')
        
        # Basic noise filtering (simple moving average)
        window_size = min(5, len(self.data) // 10)  # Adaptive window size
        if window_size > 1:
            self.data['real'] = self.data['real'].rolling(window=window_size, center=True).mean()
            self.data['imag'] = self.data['imag'].rolling(window=window_size, center=True).mean()
            # Fill NaN values at edges
            self.data['real'] = self.data['real'].fillna(method='bfill').fillna(method='ffill')
            self.data['imag'] = self.data['imag'].fillna(method='bfill').fillna(method='ffill')
            
            # Recalculate magnitude and phase
            self.data['|Z|'] = np.sqrt(self.data['real']**2 + self.data['imag']**2)
            self.data['phase'] = np.arctan2(self.data['imag'], self.data['real']) * 180 / np.pi
        
        self.logger.info("Data preprocessing completed")
    
    def _rc_element_impedance(self, f: np.ndarray, r: float, c: float) -> np.ndarray:
        """
        Calculate the impedance of an RC element.
        
        Parameters:
        -----------
        f : np.ndarray
            Frequencies in Hz
        r : float
            Resistance in Ohms
        c : float
            Capacitance in Farads
            
        Returns:
        --------
        np.ndarray
            Complex impedance values
        """
        omega = 2 * np.pi * f
        return r / (1 + 1j * omega * r * c)
    
    def _equivalent_circuit_model(self, f: np.ndarray, r_ser: float, r1: float, c1: float, 
                                r2: float, c2: float) -> np.ndarray:
        """
        Calculate the impedance of a 2RC equivalent circuit model.
        
        Parameters:
        -----------
        f : np.ndarray
            Frequencies in Hz
        r_ser : float
            Series resistance in Ohms
        r1, c1 : float
            Resistance and capacitance of first RC element
        r2, c2 : float
            Resistance and capacitance of second RC element
            
        Returns:
        --------
        np.ndarray
            Complex impedance values
        """
        z_rc1 = self._rc_element_impedance(f, r1, c1)
        z_rc2 = self._rc_element_impedance(f, r2, c2)
        
        return r_ser + z_rc1 + z_rc2
    
    def fit_equivalent_circuit(self) -> Dict[str, float]:
        """
        Fit the impedance data to a 2RC equivalent circuit model.
        
        This uses a 2RC model as described in the reference papers, which can
        represent the main electrochemical processes in a lithium-ion battery.
        
        Returns:
        --------
        Dict[str, float]
            Dictionary of fitted equivalent circuit parameters
        """
        if self.data is None:
            self.logger.error("No data to fit")
            return None
        
        # Extract frequency and impedance data
        f = self.data['frequency'].values
        z_real = self.data['real'].values
        z_imag = self.data['imag'].values
        z_measured = z_real + 1j * z_imag
        
        # Define error function for optimization
        def error_function(params):
            r_ser, r1, c1, r2, c2 = params
            z_model = self._equivalent_circuit_model(f, r_ser, r1, c1, r2, c2)
            return np.sum(np.abs(z_measured - z_model)**2)
        
        # Initial parameter guess
        initial_guess = [
            min(z_real),  # r_ser
            (max(z_real) - min(z_real)) * 0.6,  # r1
            1e-6,  # c1
            (max(z_real) - min(z_real)) * 0.4,  # r2
            1e-4   # c2
        ]
        
        # Parameter bounds
        bounds = [
            (0, None),    # r_ser > 0
            (0, None),    # r1 > 0
            (1e-9, 1e-3), # 1nF < c1 < 1mF
            (0, None),    # r2 > 0
            (1e-7, 1e-1)  # 0.1μF < c2 < 100mF
        ]
        
        try:
            # Perform optimization
            result = optimize.minimize(error_function, initial_guess, bounds=bounds, method='L-BFGS-B')
            
            if result.success:
                r_ser, r1, c1, r2, c2 = result.x
                total_resistance = r_ser + r1 + r2
                
                parameters = {
                    'R_ser': r_ser,
                    'R1': r1,
                    'C1': c1,
                    'R2': r2,
                    'C2': c2,
                    'R_total': total_resistance
                }
                
                self.equivalent_circuit_params = parameters
                self.logger.info("Successfully fitted equivalent circuit model")
                return parameters
            else:
                self.logger.warning(f"Fitting did not converge: {result.message}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error during equivalent circuit fitting: {str(e)}")
            return None
    
    def calculate_euclidean_distance(self, other_spectrum: pd.DataFrame) -> float:
        """
        Calculate the Euclidean distance between the current impedance spectrum and another one.
        This is used for pattern matching and cell grouping.
        
        Parameters:
        -----------
        other_spectrum : pd.DataFrame
            Dataframe containing another impedance spectrum with the same format
            
        Returns:
        --------
        float
            Euclidean distance between the two spectra
        """
        if self.data is None:
            self.logger.error("No data available for comparison")
            return None
            
        # Ensure both spectra have the same frequencies
        common_freqs = np.intersect1d(self.data['frequency'].values, other_spectrum['frequency'].values)
        if len(common_freqs) == 0:
            self.logger.error("No common frequencies between spectra")
            return None
            
        # Extract data for common frequencies
        self_real = np.interp(common_freqs, self.data['frequency'].values, self.data['real'].values)
        self_imag = np.interp(common_freqs, self.data['frequency'].values, self.data['imag'].values)
        
        other_real = np.interp(common_freqs, other_spectrum['frequency'].values, other_spectrum['real'].values)
        other_imag = np.interp(common_freqs, other_spectrum['frequency'].values, other_spectrum['imag'].values)
        
        # Calculate Euclidean distance
        distance = np.sqrt(np.sum((self_real - other_real)**2 + (self_imag - other_imag)**2))
        return distance
    
    def estimate_soh(self) -> float:
        """
        Estimate the State of Health (SOH) of the battery based on the impedance parameters.
        
        Returns:
        --------
        float
            Estimated SOH as a percentage
        """
        if self.equivalent_circuit_params is None:
            self.logger.warning("Equivalent circuit parameters not available. Running fitting first.")
            self.fit_equivalent_circuit()
            
        if self.equivalent_circuit_params is None:
            self.logger.error("Could not estimate SOH due to missing equivalent circuit parameters")
            return None
            
        # Simple SOH estimation model based on total resistance
        # In a real implementation, this would be a more sophisticated ML model
        R_total = self.equivalent_circuit_params['R_total']
        
        # Simplified model: SOH decreases as R_total increases
        # Assuming a fresh cell has R_total of around 5-15 mOhm for a typical Li-ion cell
        # and an end-of-life cell might have 2-3x that value
        R_fresh = 0.015  # 15 mOhm for a fresh cell (this would be calibrated)
        R_eol = 0.045    # 45 mOhm for end-of-life (this would be calibrated)
        
        soh = 100 * max(0, min(1, (R_eol - R_total) / (R_eol - R_fresh)))
        self.soh_estimate = soh
        
        self.logger.info(f"Estimated SOH: {soh:.1f}%")
        return soh
    
    def predict_rul(self, usage_profile: str = 'normal') -> Dict[str, Union[float, str]]:
        """
        Predict the Remaining Useful Life (RUL) of the battery.
        
        Parameters:
        -----------
        usage_profile : str
            Battery usage profile: 'light', 'normal', or 'heavy'
            
        Returns:
        --------
        Dict[str, Union[float, str]]
            Dictionary containing RUL prediction and confidence level
        """
        if self.soh_estimate is None:
            self.logger.warning("SOH estimate not available. Estimating SOH first.")
            self.estimate_soh()
            
        if self.soh_estimate is None:
            self.logger.error("Could not predict RUL due to missing SOH estimate")
            return None
            
        # Define degradation rates based on usage profile (percentage points per cycle)
        degradation_rates = {
            'light': 0.01,    # 0.01% SOH loss per cycle
            'normal': 0.02,   # 0.02% SOH loss per cycle
            'heavy': 0.04     # 0.04% SOH loss per cycle
        }
        
        if usage_profile not in degradation_rates:
            self.logger.warning(f"Unknown usage profile: {usage_profile}. Using 'normal' instead.")
            usage_profile = 'normal'
            
        # Calculate remaining cycles
        end_of_life_soh = 80.0  # Typically, 80% SOH is considered end-of-life for EVs
        remaining_soh = self.soh_estimate - end_of_life_soh
        
        if remaining_soh <= 0:
            cycles_remaining = 0
            confidence = "high"
        else:
            degradation_rate = degradation_rates[usage_profile]
            cycles_remaining = remaining_soh / degradation_rate
            
            # Determine confidence level based on SOH
            if self.soh_estimate > 90:
                confidence = "medium"  # Harder to predict far in advance
            elif self.soh_estimate > 85:
                confidence = "high"
            else:
                confidence = "very high"  # More accurate near end-of-life
                
        # Convert cycles to approximate time
        cycles_per_month = {'light': 15, 'normal': 30, 'heavy': 60}
        months_remaining = cycles_remaining / cycles_per_month[usage_profile]
        
        result = {
            'remaining_cycles': cycles_remaining,
            'remaining_months': months_remaining,
            'confidence': confidence,
            'usage_profile': usage_profile
        }
        
        self.rul_prediction = result
        self.logger.info(f"Predicted RUL: {cycles_remaining:.0f} cycles ({months_remaining:.1f} months) with {confidence} confidence")
        
        return result
    
    def detect_anomalies(self) -> List[Dict[str, Union[str, float]]]:
        """
        Detect anomalies in the impedance spectrum that might indicate potential issues.
        
        Returns:
        --------
        List[Dict[str, Union[str, float]]]
            List of detected anomalies with type and severity
        """
        if self.data is None or self.equivalent_circuit_params is None:
            self.logger.error("Cannot detect anomalies without data and equivalent circuit parameters")
            return []
            
        anomalies = []
        
        # Check for unusually high series resistance
        if self.equivalent_circuit_params['R_ser'] > 0.03:  # 30 mOhm threshold
            anomalies.append({
                'type': 'high_series_resistance',
                'severity': 'medium' if self.equivalent_circuit_params['R_ser'] < 0.05 else 'high',
                'value': self.equivalent_circuit_params['R_ser'],
                'description': 'Unusually high series resistance may indicate connection issues or current collector corrosion'
            })
            
        # Check for unusually high charge transfer resistance (R1)
        if self.equivalent_circuit_params['R1'] > 0.05:  # 50 mOhm threshold
            anomalies.append({
                'type': 'high_charge_transfer_resistance',
                'severity': 'medium' if self.equivalent_circuit_params['R1'] < 0.1 else 'high',
                'value': self.equivalent_circuit_params['R1'],
                'description': 'High charge transfer resistance may indicate SEI layer growth or electrode degradation'
            })
            
        # Check for unusually low double layer capacitance (C1)
        if self.equivalent_circuit_params['C1'] < 1e-7:  # 0.1 µF threshold
            anomalies.append({
                'type': 'low_double_layer_capacitance',
                'severity': 'medium',
                'value': self.equivalent_circuit_params['C1'],
                'description': 'Low double layer capacitance may indicate loss of active surface area'
            })
            
        # Check for unusual spectrum shape using pattern analysis
        # This would be more sophisticated in a real implementation
        
        self.logger.info(f"Detected {len(anomalies)} anomalies")
        return anomalies
    
    def analyze(self) -> Dict[str, any]:
        """
        Perform a comprehensive analysis of the impedance data.
        
        Returns:
        --------
        Dict[str, any]
            Dictionary containing all analysis results
        """
        results = {}
        
        # Fit equivalent circuit model
        results['equivalent_circuit'] = self.fit_equivalent_circuit()
        
        # Estimate State of Health
        results['soh'] = self.estimate_soh()
        
        # Predict Remaining Useful Life
        results['rul'] = self.predict_rul()
        
        # Detect anomalies
        results['anomalies'] = self.detect_anomalies()
        
        # Calculate cell grouping score (can be used for battery pack assembly)
        # This would be based on the equivalent circuit parameters
        if self.equivalent_circuit_params is not None:
            # Calculate a grouping score based on total resistance
            # Cells with similar scores should be grouped together
            R_total = self.equivalent_circuit_params['R_total']
            results['cell_grouping_score'] = R_total * 1000  # Convert to mOhm for easier reading
        
        self.logger.info("Analysis completed successfully")
        return results
    
    def plot_nyquist(self, fitted: bool = True, save_path: Optional[str] = None) -> None:
        """
        Create a Nyquist plot of the impedance data.
        
        Parameters:
        -----------
        fitted : bool
            Whether to plot the fitted equivalent circuit model alongside measured data
        save_path : Optional[str]
            Path to save the plot. If None, the plot is displayed but not saved.
        """
        if self.data is None:
            self.logger.error("No data to plot")
            return
            
        plt.figure(figsize=(10, 8))
        
        # Plot measured data
        plt.plot(self.data['real'], -self.data['imag'], 'o', markersize=5, label='Measured')
        
        # Plot fitted model if requested and available
        if fitted and self.equivalent_circuit_params is not None:
            f = np.logspace(np.log10(min(self.data['frequency'])), 
                           np.log10(max(self.data['frequency'])), 
                           100)
            
            r_ser = self.equivalent_circuit_params['R_ser']
            r1 = self.equivalent_circuit_params['R1']
            c1 = self.equivalent_circuit_params['C1']
            r2 = self.equivalent_circuit_params['R2']
            c2 = self.equivalent_circuit_params['C2']
            
            z_model = self._equivalent_circuit_model(f, r_ser, r1, c1, r2, c2)
            plt.plot(z_model.real, -z_model.imag, '-', linewidth=2, label='Fitted Model')
            
        plt.xlabel('Z\' (Ω)', fontsize=12)
        plt.ylabel('-Z\'\' (Ω)', fontsize=12)
        plt.title('Nyquist Plot', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.axis('equal')  # Equal aspect ratio
        
        # Add circuit parameters if available
        if self.equivalent_circuit_params is not None:
            param_text = (
                f"R_ser = {self.equivalent_circuit_params['R_ser']*1000:.2f} mΩ\n"
                f"R1 = {self.equivalent_circuit_params['R1']*1000:.2f} mΩ\n"
                f"C1 = {self.equivalent_circuit_params['C1']*1e6:.2f} µF\n"
                f"R2 = {self.equivalent_circuit_params['R2']*1000:.2f} mΩ\n"
                f"C2 = {self.equivalent_circuit_params['C2']*1e6:.2f} µF\n"
                f"R_total = {self.equivalent_circuit_params['R_total']*1000:.2f} mΩ"
            )
            plt.annotate(param_text, xy=(0.02, 0.02), xycoords='axes fraction', 
                         bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        # Add SOH if available
        if self.soh_estimate is not None:
            soh_text = f"SOH: {self.soh_estimate:.1f}%"
            plt.annotate(soh_text, xy=(0.75, 0.95), xycoords='axes fraction', 
                         bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Nyquist plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_bode(self, save_path: Optional[str] = None) -> None:
        """
        Create Bode plots (magnitude and phase vs. frequency) of the impedance data.
        
        Parameters:
        -----------
        save_path : Optional[str]
            Path to save the plot. If None, the plot is displayed but not saved.
        """
        if self.data is None:
            self.logger.error("No data to plot")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Magnitude plot
        ax1.semilogx(self.data['frequency'], self.data['|Z|'], 'o-', markersize=5)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('|Z| (Ω)')
        ax1.set_title('Impedance Magnitude')
        ax1.grid(True, which="both", alpha=0.3)
        
        # Phase plot
        ax2.semilogx(self.data['frequency'], self.data['phase'], 'o-', markersize=5)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (°)')
        ax2.set_title('Impedance Phase')
        ax2.grid(True, which="both", alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Bode plot saved to {save_path}")
        else:
            plt.show()
    
    def generate_report(self, results: Dict[str, any], output_path: str) -> None:
        """
        Generate a comprehensive report of the battery analysis.
        
        Parameters:
        -----------
        results : Dict[str, any]
            Analysis results from the analyze() method
        output_path : str
            Path to save the report
        """
        try:
            # This is a simplified version - in a real implementation, 
            # this would generate a proper PDF or HTML report
            with open(output_path, 'w') as f:
                f.write("# Battery Impedance Analysis Report\n\n")
                
                # Battery State of Health
                f.write("## State of Health\n")
                f.write(f"Estimated SOH: {results.get('soh', 'N/A')}%\n\n")
                
                # Remaining Useful Life
                if 'rul' in results and results['rul']:
                    f.write("## Remaining Useful Life\n")
                    f.write(f"Estimated remaining cycles: {results['rul'].get('remaining_cycles', 'N/A')}\n")
                    f.write(f"Estimated remaining months: {results['rul'].get('remaining_months', 'N/A')}\n")
                    f.write(f"Confidence: {results['rul'].get('confidence', 'N/A')}\n")
                    f.write(f"Usage profile: {results['rul'].get('usage_profile', 'N/A')}\n\n")
                
                # Equivalent Circuit Parameters
                if 'equivalent_circuit' in results and results['equivalent_circuit']:
                    f.write("## Equivalent Circuit Parameters\n")
                    f.write(f"Series resistance (R_ser): {results['equivalent_circuit'].get('R_ser', 'N/A')*1000:.2f} mΩ\n")
                    f.write(f"First RC element resistance (R1): {results['equivalent_circuit'].get('R1', 'N/A')*1000:.2f} mΩ\n")
                    f.write(f"First RC element capacitance (C1): {results['equivalent_circuit'].get('C1', 'N/A')*1e6:.2f} µF\n")
                    f.write(f"Second RC element resistance (R2): {results['equivalent_circuit'].get('R2', 'N/A')*1000:.2f} mΩ\n")
                    f.write(f"Second RC element capacitance (C2): {results['equivalent_circuit'].get('C2', 'N/A')*1e6:.2f} µF\n")
                    f.write(f"Total resistance (R_total): {results['equivalent_circuit'].get('R_total', 'N/A')*1000:.2f} mΩ\n\n")
                
                # Anomalies
                if 'anomalies' in results and results['anomalies']:
                    f.write("## Detected Anomalies\n")
                    for anomaly in results['anomalies']:
                        f.write(f"- {anomaly.get('type', 'Unknown anomaly')}: {anomaly.get('description', '')}\n")
                        f.write(f"  Severity: {anomaly.get('severity', 'unknown')}\n")
                    f.write("\n")
                
                # Cell Grouping
                if 'cell_grouping_score' in results:
                    f.write("## Cell Grouping\n")
                    f.write(f"Cell grouping score: {results['cell_grouping_score']:.2f}\n")
                    f.write("This cell should be grouped with others having similar scores for optimal battery pack performance.\n\n")
                
                # Recommendations
                f.write("## Recommendations\n")
                
                if results.get('soh', 0) < 80:
                    f.write("- Consider replacing the battery as it has reached end of life criteria.\n")
                elif results.get('soh', 0) < 85:
                    f.write("- Battery is approaching end of life. Plan for replacement in the near future.\n")
                else:
                    f.write("- Battery health is good. Continue normal operation.\n")
                
                if 'anomalies' in results and any(a.get('severity') == 'high' for a in results['anomalies']):
                    f.write("- High severity anomalies detected. Immediate attention recommended.\n")
                
                f.write("\n")
                
                # Report generation metadata
                import datetime
                f.write(f"Report generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            self.logger.info(f"Report generated successfully and saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise
