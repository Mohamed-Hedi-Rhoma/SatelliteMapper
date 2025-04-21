import numpy as np

class RSRReader:
    def __init__(self, file_path):
        """
        Initialize the RSR reader with a file path.
        
        Parameters:
        -----------
        file_path : str
            Path to the RSR file
        """
        self.file_path = file_path
        
    def read_rsr_file(self):
        """
        Read the RSR file and extract three matrices:
        1. Full matrix with all 6 bands
        2. Matrix with just the 4 visible/NIR bands (blue, green, red, NIR), zeros removed
        3. Matrix with just the 2 SWIR bands (SWIR1, SWIR2), zeros removed
        
        Returns:
        --------
        full_matrix : numpy.ndarray
            Matrix of shape [6, N] with all band sensitivities
        vnir_matrix : numpy.ndarray
            Matrix of shape [4, M] with blue, green, red, NIR band sensitivities (zeros removed)
        swir_matrix : numpy.ndarray
            Matrix of shape [2, K] with SWIR1, SWIR2 band sensitivities (zeros removed)
        wavelengths : numpy.ndarray
            Array of shape [N] with the wavelengths
        vnir_wavelengths : numpy.ndarray
            Array of shape [M] with wavelengths for VNIR bands
        swir_wavelengths : numpy.ndarray
            Array of shape [K] with wavelengths for SWIR bands
        """
        # Read all data from the file
        data = []
        with open(self.file_path, 'r') as f:
            for line in f:
                values = line.strip().split()
                if len(values) >= 15:  # Ensure we have enough columns
                    data.append([float(v) for v in values])
        
        # Convert to numpy array for easier manipulation
        data_array = np.array(data)
        
        # Extract wavelengths (first column)
        wavelengths = data_array[:, 0]
        
        # Count the number of wavelengths
        n_wavelengths = data_array.shape[0]
        
        # Create the output full matrix
        full_matrix = np.zeros((6, n_wavelengths))
        
        # Define the column indices for each band based on the analysis
        # Column indices are 0-based in the array (subtract 1 from the file columns)
        # Blue: column 3, Green: column 4, Red: column 5, 
        # NIR: column 9, SWIR1: column 13, SWIR2: column 14
        band_indices = [2, 3, 4, 8, 12, 13]
        
        # Extract values for each band into the full matrix
        for i, band_idx in enumerate(band_indices):
            full_matrix[i] = data_array[:, band_idx]
        
        # Create temporary VNIR and SWIR matrices
        temp_vnir_matrix = full_matrix[:4]
        temp_swir_matrix = full_matrix[4:]
        
        # Find indices where at least one VNIR band has non-zero value
        vnir_non_zero_indices = np.where(np.sum(temp_vnir_matrix, axis=0) > 0)[0]
        
        # Find indices where at least one SWIR band has non-zero value
        swir_non_zero_indices = np.where(np.sum(temp_swir_matrix, axis=0) > 0)[0]
        
        # Create VNIR matrix with zeros removed
        vnir_matrix = temp_vnir_matrix[:, vnir_non_zero_indices]
        vnir_wavelengths = wavelengths[vnir_non_zero_indices]
        
        # Create SWIR matrix with zeros removed
        swir_matrix = temp_swir_matrix[:, swir_non_zero_indices]
        swir_wavelengths = wavelengths[swir_non_zero_indices]
        
        return full_matrix, vnir_matrix, swir_matrix, wavelengths, vnir_wavelengths, swir_wavelengths

# Example usage
if __name__ == "__main__":
    rsr_reader = RSRReader("C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/pth_data/sentinel2.rsr")
    full_matrix, vnir_matrix, swir_matrix, wavelengths, vnir_wavelengths, swir_wavelengths = rsr_reader.read_rsr_file()
    
    # Print the shapes of the resulting matrices
    print(f"Full wavelengths shape: {wavelengths.shape}")
    print(f"Full matrix shape: {full_matrix.shape}")
    print(f"VNIR wavelengths shape: {vnir_wavelengths.shape}")
    print(f"VNIR matrix shape: {vnir_matrix.shape}")
    print(f"SWIR wavelengths shape: {swir_wavelengths.shape}")
    print(f"SWIR matrix shape: {swir_matrix.shape}")
    
    # Print the wavelength ranges using "um" instead of the Greek mu symbol
    print(f"\nFull wavelength range: {wavelengths[0]} - {wavelengths[-1]} um")
    print(f"VNIR wavelength range: {vnir_wavelengths[0]} - {vnir_wavelengths[-1]} um")
    print(f"SWIR wavelength range: {swir_wavelengths[0]} - {swir_wavelengths[-1]} um")
    
    # Print the number of non-zero entries
    print(f"\nPercentage of non-zero wavelengths in VNIR: {100 * len(vnir_wavelengths) / len(wavelengths):.2f}%")
    print(f"Percentage of non-zero wavelengths in SWIR: {100 * len(swir_wavelengths) / len(wavelengths):.2f}%")