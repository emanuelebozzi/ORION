class Das_h5:
    """
    Loads DAS .h5 files with support for both:
    - Original nested 'Acquisition/Raw[0]/RawData' format
    - Flat 'data' dataset format
    """

    def __init__(self, file_name, file_format, data_format='strain', unwrap=False):
        if file_format != 'h5':
            raise ValueError("Only 'h5' file format is supported.")

        import h5py
        import numpy as np

        with h5py.File(file_name, 'r') as f:
            # List datasets in the file
            datasets = self._list_datasets(f)

            if "Acquisition/Raw[0]/RawData" in datasets:
                # Original nested format
                self._load_original_format(f, data_format)
            elif "data" in datasets:
                # Flat format
                self._load_flat_format(f, data_format)
            else:
                raise ValueError(
                    f"Unsupported .h5 structure. Found datasets: {datasets}"
                )

    def _list_datasets(self, group, prefix=""):
        """Recursively list all dataset paths in the HDF5 file."""
        import h5py
        paths = []
        for key, item in group.items():
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(item, h5py.Dataset):
                paths.append(path)
            elif isinstance(item, h5py.Group):
                paths.extend(self._list_datasets(item, path))
        return paths

    def _load_original_format(self, f, data_format):
        """Load the original nested Acquisition format."""
        import numpy as np

        self.pulserate = f['Acquisition'].attrs['PulseRate']
        self.channel_spacing = f['Acquisition'].attrs['SpatialSamplingInterval']
        self.dx = self.channel_spacing

        self.data = f['Acquisition']['Raw[0]']['RawData'][()]
        self.gaugeLength = f['Acquisition'].attrs['GaugeLength']
        self.samplingRate = float(f['Acquisition']['Raw[0]'].attrs['OutputDataRate'])
        self.dt = 1 / self.samplingRate

        self.FibreRefractiveIndex = f['Acquisition']['Custom'].attrs['Fibre Refractive Index']
        self.tinit = f['Acquisition']['Raw[0]']['RawData'].attrs['PartStartTime']
        self.tend = f['Acquisition']['Raw[0]']['RawData'].attrs['PartEndTime']

        self.RawDataUnit = f['Acquisition']['Raw[0]'].attrs['RawDataUnit']
        self.data = self.data.astype(float)
        self.photo_elastic_coeff = 0.78

        self._convert_units(data_format)

    def _load_flat_format(self, f, data_format):
        """Load flat format with a single 'data' dataset."""
        import numpy as np

        self.data = f['data'][()].astype(float)
        self.photo_elastic_coeff = 0.78

        # Attributes may not exist, so use safe defaults
        self.pulserate = f.attrs.get('PulseRate', None)
        self.channel_spacing = f.attrs.get('SpatialSamplingInterval', None)
        self.dx = self.channel_spacing
        self.gaugeLength = f.attrs.get('GaugeLength', 1.0)
        self.samplingRate = f.attrs.get('OutputDataRate', None)
        self.dt = 1 / self.samplingRate if self.samplingRate else None
        self.FibreRefractiveIndex = f.attrs.get('Fibre Refractive Index', 1.45)
        self.tinit = f.attrs.get('PartStartTime', None)
        self.tend = f.attrs.get('PartEndTime', None)
        self.RawDataUnit = f.attrs.get('RawDataUnit', None)

        self._convert_units(data_format)

    def _convert_units(self, data_format):
        """Convert raw data to strain if requested."""
        import numpy as np

        to_rad = (2 * np.pi) / (2**16)
        self.data *= to_rad

        if data_format == 'strain':
            opt_wavelength = 1550.12 * 1e-9
            rads_to_strain = opt_wavelength / (
                4 * np.pi * self.photo_elastic_coeff *
                self.gaugeLength * self.FibreRefractiveIndex
            )
            self.data *= rads_to_strain
            self.data *= 1e6  # microstrain

        self.data = np.transpose(self.data)
