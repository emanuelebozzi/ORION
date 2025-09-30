# Standard Libraries
import os
import glob
import shutil
import csv
import random

# Numerical & Data Handling
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
import seaborn as sns
from mpl_toolkits.basemap import Basemap

# Signal Processing
from scipy.signal import detrend, tukey, butter, filtfilt, correlate, find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import griddata

# Machine Learning
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

# Geospatial
from geopy.distance import geodesic

# ObsPy (Seismic Data)
from obspy import read, Stream, Trace, UTCDateTime
from obspy.signal.trigger import classic_sta_lta, trigger_onset

# Utilities
import joblib

# Custom modules
from read_h5 import Das_h5

# Progress bars
from tqdm import tqdm

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",   # or "Liberation Sans" / "Ubuntu"
    "font.weight": "light",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "lines.linewidth": 1,
    "lines.markersize": 4,
    "axes.linewidth": 1.5,   # <<< Thicker axis edge (default ~0.8)
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
})





class DASChannelSelector:
    def __init__(
        self,
        name: str,
        geometry_path: str,
        das_data_path: str,
        output_dir: str = "output"  # Default to 'output' directory
    ):
        self.name = name
        self.geometry_path = geometry_path
        self.das_data_path = das_data_path
        self.output_dir = output_dir

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)


    def dms_to_decimal(self, dms_str):
        # Dummy converter; replace with actual if needed
        dms = dms_str.strip().replace("°", " ").replace("'", " ").replace('"', ' ')
        parts = list(map(float, dms.split()))
        sign = -1 if "-" in dms_str else 1
        return sign * (parts[0] + parts[1] / 60 + parts[2] / 3600)

    def simplify_geometry_csv(self, filename="simplified_geometry.csv"):
        """
        Reads a full CSV (with DMS or decimal coordinates), simplifies it,
        and saves a new CSV with only channel_id, longitude, and latitude
        inside self.output_dir.

        If a 'das_id' field is present, it will be used as channel_id.
        """
        import os
        output_path = os.path.join(self.output_dir, filename)

        # Detect delimiter
        with open(self.geometry_path, 'r', encoding='utf-8') as f:
            sample = f.read(2048)
            try:
                sep = csv.Sniffer().sniff(sample).delimiter
            except csv.Error:
                sep = ';'

        df = pd.read_csv(self.geometry_path, sep=sep)
        df.columns = [c.strip() for c in df.columns]

        # Detect coordinate format
        if df['latitude'].astype(str).str.contains('[°\'"]', regex=True).any():
            df['latitude'] = df['latitude'].apply(self.safe_dms_to_decimal)
            df['longitude'] = df['longitude'].apply(self.safe_dms_to_decimal)

        df = df.dropna(subset=['latitude', 'longitude'])

        # Handle channel_id logic
        if 'das_id' in df.columns:
            df['channel_id'] = df['das_id']
        elif 'channel_id' not in df.columns:
            df['channel_id'] = np.arange(len(df))

        # Save simplified
        simplified = df[['channel_id', 'longitude', 'latitude']].copy()
        os.makedirs(self.output_dir, exist_ok=True)
        simplified.to_csv(output_path, index=False)
        print(f"Simplified geometry saved to {output_path}")



    def safe_dms_to_decimal(self, val):
        if pd.isna(val) or not isinstance(val, str):
            return np.nan
        try:
            return self.dms_to_decimal(val)
        except Exception:
            return np.nan

    def load_geometry(self, output_path):
        """
        Loads geometry from a simplified CSV with columns: channel_id, longitude, latitude.
        """
        try:
            df = pd.read_csv(output_path)
        except Exception as e:
            raise ValueError(f"Could not read simplified geometry CSV: {e}")

        df.columns = [c.strip().lower() for c in df.columns]

        required_cols = {'channel_id', 'latitude', 'longitude'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Simplified CSV must include columns: {required_cols}")

        df = df.dropna(subset=['latitude', 'longitude'])

        # Save to class
        self.df = df
        self.lat = df['latitude'].to_numpy()
        self.lon = df['longitude'].to_numpy()
        self.depth = np.zeros(len(df))  # Depth not present in simplified
        self.azimuth = np.zeros(len(df))  # Azimuth not present
        self.das_id = df['channel_id'].to_numpy()



    def plot_geometry(self):
        if self.df is None:
            raise RuntimeError("Geometry not loaded. Call load_geometry() first.")

        # Set up latitude and longitude ranges
        margin = 0.05

        das_id = self.das_id

        lat_range = (self.lat.min() - margin, self.lat.max() + margin)
        lon_range = (self.lon.min() - margin, self.lon.max() + margin)

        # Create figure
        fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

        # Setup Basemap
        m = Basemap(projection='cyl', llcrnrlat=lat_range[0], urcrnrlat=lat_range[1],
                    llcrnrlon=lon_range[0], urcrnrlon=lon_range[1], resolution='h', ax=ax)
        m.drawcoastlines()
        m.drawcountries()
        m.fillcontinents(color='lightgray', lake_color='lightblue')
        m.drawmapboundary(fill_color='lightblue')

        # Grid lines
        parallels = np.arange(lat_range[0], lat_range[1] + 0.01, 0.05)
        meridians = np.arange(lon_range[0], lon_range[1] + 0.01, 0.05)
        m.drawparallels(parallels, labels=[1,0,0,0], linewidth=0.3, dashes=[1,1], color='gray')
        m.drawmeridians(meridians, labels=[0,0,0,1], linewidth=0.3, dashes=[1,1], color='gray')

        ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

        # Scatter points
        x = self.lon
        y = self.lat
        sc = ax.scatter(x, y, marker='^', c=das_id, s=25, zorder=3, label='DAS array')

        # Colorbar
        cbar = plt.colorbar(sc, ax=ax, shrink=0.4)
        cbar.set_label('DAS ID')

        # Title and legend
        ax.set_title('DAS Array')
        plt.legend()
        plt.tight_layout()

        # Save to output directory
        output_path = os.path.join(self.output_dir, "das_array_plot.pdf")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()


    def __design_iir_filter(self,cutoff, filter_type, order=4):
        from scipy.signal import iirfilter, sosfiltfilt, sosfilt, zpk2sos
        sampling_rate= self.sampling_rate
        nyquist = 0.5 * sampling_rate  # Nyquist frequency
        if filter_type == 'lowpass' or filter_type == 'highpass':
            normal_cutoff = cutoff / nyquist
        elif filter_type == 'bandpass':
            normal_cutoff = [c / nyquist for c in cutoff]  # Normalize for bandpass
        else:
            raise ValueError("Invalid filter type. Use 'lowpass', 'highpass', or 'bandpass'.")
        # Design IIR filter using iirfilter
        z, p, k = iirfilter(order, normal_cutoff, btype=filter_type, ftype='butter', output='zpk')
        # Convert to second-order sections (SOS) for numerical stability
        sos = zpk2sos(z, p, k)
        return sos            


    def filter(self,data, filter_type, cutoff_freqs, order=4, zerophase=True):
        from scipy.signal import iirfilter, sosfiltfilt, sosfilt, zpk2sos
        """
        Apply an IIR filter (lowpass, highpass, or bandpass) using zpk2sos and iirfilter.

        Parameters:
        - data: 2D numpy array where each row is a time series (shape (n_rows, n_samples)).
        - filter_type: Type of filter ('lowpass', 'highpass', 'bandpass').
        - cutoff_freqs: Critical frequencies for the filter. Should be:
          - A single float for 'lowpass' and 'highpass'.
          - A tuple of two floats for 'bandpass' (low_cutoff, high_cutoff).
        - sampling_rate: The sampling rate of the data (in Hz).
        - order: The order of the filter (default is 4).
        - zerophase: Whether to apply zero-phase filtering (default is True, uses sosfiltfilt for zero-phase filtering).

        Returns:
        - filtered_data: The filtered data as a 2D numpy array.
        """
        
        # Design the IIR filter using iirfilter and convert to SOS form
        sos = self.__design_iir_filter(cutoff_freqs, filter_type, order)
        
        # Apply the filter row-wise (to each time series)
        if zerophase:
            # Zero-phase filtering using sosfiltfilt
            filtered_data = np.array([sosfiltfilt(sos, row) for row in data])
        else:
            # Regular filtering using sosfilt (causal)
            filtered_data = np.array([sosfilt(sos, row) for row in data])

        return filtered_data



    def read_das_data(
        self,
        sampling_rate: int = 500,
        detrend_data: bool = True,
        apply_filter: bool = True,
        apply_taper: bool = True,
        taper_alpha: float = 0.01,
        low_freq: float = 2.0,
        high_freq: float = 30.0,
        dx: float = 2.0, 
        remove_k0: bool = False
    ):
        steps = [
            "Loading data",
            "Detrending",
            "Applying taper",
            "Applying filter",
            "Removing k0 frequency (FK filter)" if remove_k0 else None,
            "Finished"
        ]



        def trace_normalization(data, demean=False, method="max", eps=1e-12):
            """
            Normalize traces in a 2D array.

            Parameters
            ----------
            data : np.ndarray
                2D array with shape (n_traces, n_points).
            demean : bool, optional
                If True, subtract mean from each trace before normalization.
            method : str, optional
                Normalization method:
                - "max" : divide by maximum absolute value (default)
                - "rms" : divide by root mean square value
            eps : float, optional
                Small value to avoid division by zero or tiny numbers.

            Returns
            -------
            np.ndarray
                Normalized data, same shape as input.
            """
            data = np.array(data, copy=True)  # work on a copy to avoid modifying input
            ntrs, npts = np.shape(data)

            for i in range(ntrs):
                trace = data[i, :]

                # Demean if requested
                if demean:
                    trace = trace - np.mean(trace)

                # Choose normalization method
                if method == "max":
                    nf = np.max(np.abs(trace))
                elif method == "rms":
                    nf = np.sqrt(np.mean(trace**2))
                else:
                    raise ValueError("method must be 'max' or 'rms'")

                # Avoid division by zero / tiny numbers
                nf = max(nf, eps)

                data[i, :] = trace / nf

            return data

        

        steps = [step for step in steps if step is not None]
        pbar = tqdm(total=len(steps), desc="read_das_data", unit="step")

        ext = os.path.splitext(self.das_data_path)[-1].lower()

        if ext == ".npy":
            data = np.load(self.das_data_path, allow_pickle=True)
            self.sampling_rate = sampling_rate
            self.dx = dx

        elif ext in [".mseed", ".miniseed"]:
            st = read(self.das_data_path)
            st.merge(method=1, fill_value='interpolate')
            data = np.stack([tr.data.astype(np.float32) for tr in st])
            self.sampling_rate = sampling_rate

        elif ext == ".h5":
            das_obj = Das_h5(file_name=self.das_data_path, file_format='h5', data_format='strain')
            data = das_obj.data
            self.sampling_rate = sampling_rate
            self.dx = dx     # optionally store for later
        elif ext == ".dat":
            raise NotImplementedError("Add .dat file support if needed.")
        else:
            raise ValueError(f"Unsupported DAS file format: {ext}")

        data = data.T

        print(f"Loaded DAS data → shape: {data.shape}")

        pbar.update(1)  # Loading data done

        plt.figure()
        plt.plot(data[100, :], color='k', lw=0.5)
        plt.title('Example raw trace')

        # Save the figure
        plt.savefig("example_raw_trace.pdf", dpi=300, bbox_inches='tight')

        # (optional) Show the figure
        plt.close()

        data = trace_normalization(data, demean=False, method="rms")

        if detrend_data:
            data = detrend(data, axis=1, type='constant')
            data = detrend(data, axis=1, type='linear')

            plt.figure()
            plt.plot(data[100, :], color='k', lw=0.5)
            plt.title('Example raw normalized trace')

            # Save the figure
            plt.savefig("example_raw_normalized_trace.pdf", dpi=300, bbox_inches='tight')

            # (optional) Show the figure
            plt.close()
        pbar.update(1)

        if apply_taper:
            window = tukey(data.shape[1], alpha=taper_alpha)
            data *= window[np.newaxis, :]

            #data=data-np.mean(data)
            #ntrs,npts = np.shape(data)
            #for i in range(ntrs):
            #    nf=np.max(np.abs(data[i,:]))
            #    data[i,:]=data[i,:]/nf
        pbar.update(1)

        if apply_filter:


            nyquist = 0.5 * self.sampling_rate
            low = low_freq / nyquist
            high = high_freq / nyquist
            b, a = butter(N=4, Wn=[low, high], btype='bandpass')
            data = filtfilt(b, a, data, axis=1)


        pbar.update(1)

        if remove_k0 and data.shape[0] > 1:
            self.dt = 1 / self.sampling_rate
            self.ntrs, self.npts = data.shape

            f = np.fft.fftfreq(self.npts, d=self.dt)
            k = np.fft.fftfreq(self.ntrs, d=self.dx)

            fk = np.fft.fft2(data)
            n, m = fk.shape

            # Gaussian low-k suppression filter
            k_vals = np.fft.fftfreq(n, d=self.dx)
            K = np.tile(k_vals[:, None], (1, m))
            filt = 1 - np.exp(-(K / 0.01) ** 2)  # adjust 0.01 to control cutoff

            fkfilt = fk * filt
            data_filt = np.fft.ifft2(fkfilt).real


            self.das_data = trace_normalization(data, demean=False, method="rms")

            return self.das_data

        else:
            pbar.update(1)

        self.das_data = data

        plt.figure()
        plt.plot(self.das_data[100, :], color='k', lw=0.5)
        plt.title('Example filtered trace')

        # Save the figure
        plt.savefig("example_filtered_trace.pdf", dpi=300, bbox_inches='tight')

        # (optional) Show the figure
        plt.close()


        print(f"Loaded DAS data → shape: {self.das_data.shape}")

        pbar.close()

        print(f"Loaded DAS data → shape: {self.das_data.shape}")


    def compute_average_azimuth(self, chunk_size: int = 10):
        if self.df is None:
            raise RuntimeError("Geometry not loaded.")
        
        azs, dists, centers = [], [], []

        max_index = len(self.lat) - chunk_size
        for i in range(max_index + 1):  # Sliding window
            i0, i1 = i, i + chunk_size
            p0 = (self.lat[i0], self.lon[i0])
            p1 = (self.lat[i1 - 1], self.lon[i1 - 1])
            p2 = (self.depth[i1 - 1], self.depth[i1 - 1])  # Placeholder if needed
            d = geodesic(p0, p1, p2).meters
            dists.append(d)

            dlon = np.radians(self.lon[i1 - 1] - self.lon[i0])
            la0, la1 = np.radians(self.lat[i0]), np.radians(self.lat[i1 - 1])
            x = np.sin(dlon) * np.cos(la1)
            y = np.cos(la0) * np.sin(la1) - np.sin(la0) * np.cos(la1) * np.cos(dlon)
            az = (np.degrees(np.arctan2(x, y)) + 360) % 360
            azs.append(az)
            centers.append(i + chunk_size // 2)

        self.azimuths = np.array(azs)
        self.cumulative_r = np.insert(np.cumsum(dists), 0, 0.0)[:-1]
        self.chunk_centers = np.array(centers)

        # average azimuth vectorially
        rad = np.radians(self.azimuths)
        Q, P = np.sum(np.sin(rad)), np.sum(np.cos(rad))
        self.average_azimuth_deg = (np.degrees(np.arctan2(Q, P)) + 360) % 360

        print(f"Chunks: {len(self.azimuths)}, Avg azimuth: {self.average_azimuth_deg:.2f}°")


    def estimate_dbscan_eps(self, min_samples=10, plot=True):
        """
        Automatically estimate a good DBSCAN eps using k-distance heuristic.

        Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996).
        A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise.

        Satopaa, V., Albrecht, J., Irwin, D., & Raghavan, B. (2011).
        Finding a "Kneedle" in a Haystack: Detecting Knee Points in System Behavior.
        2011 IEEE 31st International Conference on Distributed Computing Systems Workshops

        """
        if self.azimuths is None or self.cumulative_r is None:
            raise RuntimeError("Run compute_average_azimuth() first.")

        # Encode azimuth as 2D vector to respect circular nature
        az_rad = np.radians(self.azimuths)
        az_cos = np.cos(az_rad)
        az_sin = np.sin(az_rad)

        # Feature matrix: distance + azimuth components
        X = np.column_stack((self.cumulative_r, az_cos, az_sin))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        neigh = NearestNeighbors(n_neighbors=min_samples)
        nbrs = neigh.fit(X_scaled)
        distances, _ = nbrs.kneighbors(X_scaled)
        k_distances = np.sort(distances[:, -1])
        
        print("Estimating eps using k-distance heuristic...")
        
        kl = KneeLocator(range(len(k_distances)), k_distances, curve='convex', direction='increasing')
        eps = k_distances[kl.knee] if kl.knee is not None else np.percentile(k_distances, 90)


        if plot:
            plt.figure(figsize=(6, 3))
            plt.plot(k_distances)
            plt.axhline(eps, color='r', linestyle='--', label=f"Selected eps={eps:.3f}")
            plt.title('K-distance plot for DBSCAN eps selection')
            plt.xlabel('Points sorted by distance to k-th NN')
            plt.ylabel(f'Distance to {min_samples}-th NN')
            plt.legend()

            # Save instead of showing
            output_path = os.path.join(self.output_dir, f"{self.name}_k_distance_plot.pdf")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()


        print(f"Estimated eps for DBSCAN: {eps:.3f}")
        return eps
    

    def cluster_by_azimuth_distance(self, eps=0.5, min_samples=5, window_size=5):
        """
        Custom DBSCAN clustering enforcing contiguity along cumulative_r.
        
        Parameters:
        - eps: max distance in feature space (scaled).
        - min_samples: minimum number of neighbors (including self) to form a core point.
        - window_size: max step size along index to check neighbors (enforces contiguity).
        """

        # Ensure required attributes are computed before proceeding
        if self.azimuths is None or self.cumulative_r is None:
            raise ValueError("azimuths and cumulative_r must be computed first.")

        # Convert azimuths from degrees to radians for trigonometric functions
        az_rad = np.radians(self.azimuths)

        # Compute cosine of azimuths (helps wrap-around continuity between angles like 359° and 1°)
        az_cos = np.cos(az_rad)

        # Compute sine of azimuths (paired with cosine to represent angle as a vector)
        az_sin = np.sin(az_rad)

        # Combine features: cumulative distance and directional components (cosine and sine of azimuths)
        X = np.column_stack((self.cumulative_r, az_cos, az_sin))

        # Standardize the feature space so that each dimension contributes equally
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Total number of points
        n = len(X_scaled)

        # Initialize all labels to -1 (noise by default)
        labels = np.full(n, -1)

        # Track whether each point has been visited during clustering
        visited = np.zeros(n, dtype=bool)

        # Initialize the first cluster ID
        cluster_id = 0

        # Loop over each point in the dataset
        for i in range(n):
            if visited[i]:
                continue  # Skip if already processed
            visited[i] = True  # Mark point as visited

            # Gather neighbors within a fixed window along the index (enforcing local continuity)
            neighbors = []
            for j in range(max(0, i - window_size), min(n, i + window_size + 1)):
                # Check if point j is within eps distance in scaled feature space
                if np.linalg.norm(X_scaled[i] - X_scaled[j]) <= eps:
                    neighbors.append(j)

            # If not enough neighbors, treat as noise and continue
            if len(neighbors) < min_samples:
                continue

            # Start a new cluster
            labels[i] = cluster_id
            seeds = set(neighbors)
            seeds.discard(i)  # Remove current point to avoid processing it again

            # Expand the cluster
            while seeds:
                current = seeds.pop()
                if not visited[current]:
                    visited[current] = True

                    # Find neighbors of the current point within the local window
                    local_neighbors = []
                    for k in range(max(0, current - window_size), min(n, current + window_size + 1)):
                        if np.linalg.norm(X_scaled[current] - X_scaled[k]) <= eps:
                            local_neighbors.append(k)

                    # If current point is a core point, add its neighbors to seeds
                    if len(local_neighbors) >= min_samples:
                        seeds.update(local_neighbors)

                # Assign point to cluster if not already assigned
                if labels[current] == -1:
                    labels[current] = cluster_id

            # Increment cluster ID for the next potential cluster
            cluster_id += 1
        # Store the final cluster labels in the instance
        self.cluster_labels = labels

        # Count and report the number of noise points and clusters found
        n_noise = np.sum(labels == -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        print("Contiguous DBSCAN completed.")
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")

        # Compute average cluster size (channels per cluster)
        if n_clusters > 0:
            cluster_sizes = [np.sum(labels == lbl) for lbl in set(labels) if lbl != -1]
            if np.median(cluster_sizes) > 50: 
                self.avg_cluster_size = 50 
            else: 
                self.avg_cluster_size = np.median(cluster_sizes) 
            print(f"Average number of channels per cluster: {self.avg_cluster_size:.2f}")
        else:
            print("Average number of channels per cluster: N/A (no clusters found)")



    def save_clustering(self, filename="clustering_results.pkl"):
        # Save cluster labels and scaler for reuse
        to_save = {
            "cluster_labels": self.cluster_labels,
            "scaler": self.scaler,  # Store scaler as an attribute in cluster_by_azimuth_distance
            "min cluster size": self.avg_cluster_size,
            "chunk_centers": getattr(self, "chunk_centers", None)
        }
        joblib.dump(to_save, filename)
        print(f"Clustering results saved to {filename}")

    def load_clustering(self, filename="clustering_results.pkl"):
        data = joblib.load(filename)
        self.cluster_labels = data["cluster_labels"]
        self.scaler = data["scaler"]
        self.avg_cluster_size = data["min cluster size"]
        self.chunk_centers = data.get("chunk_centers", None)
        print(f"Clustering results loaded from {filename}")




    def plot_geometry_with_clusters(self):
        if self.df is None:
            raise RuntimeError("Geometry not loaded. Call load_geometry() first.")
        if self.cluster_labels is None:
            raise RuntimeError("Clusters not computed yet. Run cluster_by_azimuth_distance() first.")

        
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        import seaborn as sns
        import numpy as np
        import os

        # --- Ranges with margin ---
        # --- Ranges with margin ---
        margin = 0.01

        # original ranges
        lat_min, lat_max = self.lat.min(), self.lat.max()
        lon_min, lon_max = self.lon.min(), self.lon.max()

        # calculate spans
        lat_span = lat_max - lat_min
        lon_span = lon_max - lon_min

        # take the larger span so all points fit
        span = max(lat_span, lon_span)

        # recenter around the middle
        lat_mid = (lat_max + lat_min) / 2
        lon_mid = (lon_max + lon_min) / 2

        # apply span + margin
        lat_range = (lat_mid - span/2 - margin, lat_mid + span/2 + margin)
        lon_range = (lon_mid - span/2 - margin, lon_mid + span/2 + margin)



        # --- Figure ---
        fig, ax = plt.subplots(figsize=(7, 7), dpi=300, subplot_kw={'projection': ccrs.PlateCarree()})

        # --- Offline relief: land & ocean with simple shading ---
        ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='lightgray', zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='whitesmoke', zorder=0)
        ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='whitesmoke', zorder=0)
        ax.add_feature(cfeature.RIVERS.with_scale('10m'), facecolor='whitesmoke', zorder=1)

        # --- Coastlines & borders ---
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6, zorder=1)
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.6, zorder=1)

        # --- Set extent ---
        ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]])

        # --- Gridlines ---
        ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)

        # --- Cluster Colors ---
        unique_labels = sorted(set(self.cluster_labels))
        unique_non_noise = [lbl for lbl in unique_labels if lbl != -1]
        num_clusters = len(unique_non_noise)

        base_palette = sns.color_palette("tab20c", 20)

        def transform_color(color, cycle):
            arr = np.array(color)
            if cycle == 0:
                return color
            elif cycle == 1:
                return sns.desaturate(color, 0.5)
            elif cycle == 2:
                return tuple(np.clip(arr * 0.6, 0, 1))
            elif cycle == 3:
                return tuple(0.3 + 0.7 * arr)
            elif cycle == 4:
                return tuple(np.clip(arr * 1.2, 0, 1))
            else:
                return transform_color(color, cycle % 5)

        extended_palette = []
        for i in range(num_clusters):
            base_color = base_palette[i % 20]
            cycle = i // 20
            extended_palette.append(transform_color(base_color, cycle))

        cmap = ListedColormap(extended_palette)
        norm = BoundaryNorm(unique_non_noise + [max(unique_non_noise)+1], cmap.N)

        # --- Ensure lat/lon correspond to cluster_labels ---
        lat = self.lat[:len(self.cluster_labels)]
        lon = self.lon[:len(self.cluster_labels)]

        ax.scatter(lon, lat, s=35, c='grey', label='DAS channels', marker='^', facecolors='none',linewidth=0.2, zorder=5)

        # --- Plot clusters and noise ---
        for lbl in unique_labels:
            mask = self.cluster_labels == lbl
            if lbl == -1:
                ax.scatter(lon[mask], lat[mask], c='black', marker='x', s=35, label='DAS channels not-clustered (noise)', zorder=5)
            else:
                
                ax.scatter(lon[mask], lat[mask], c=[cmap(norm(lbl))], s=35, marker='^', facecolors='none',linewidth=0.2, zorder=5)

        # --- Discrete colorbar ---
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.15, ticks=unique_non_noise)
        cbar.ax.set_yticklabels([str(lbl) for lbl in unique_non_noise])
        cbar.set_label("Cluster ID")

        # --- Title and labels ---
        ax.set_title("Spatial clustering")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # --- Noise legend only ---
        #ax.legend(loc="upper right")

        plt.tight_layout()

        # --- Save as vector graphic ---
        output_path = os.path.join(self.output_dir, f"{self.name}_spatial_clustering.pdf")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()


    def analyze_waveforms_and_select_best(
        self,
        gauge_length=10,
        subsection_size: int = 20,
        win_snr=0.5,
        step_snr=0.1,
        win=2,
        start_event=None,
        percentile=50,
        sta_window_sec=2.0, 
        lta_window_sec=10.0, 
        sta_lta_on=2.5, 
        sta_lta_off=1.0,
        snr_score_weight=0.5,
        coherence_score_weight=0.5,
        noise_rms_weight=0.2,
        n_final_select: int = 10,   # <<< NEW PARAM
        min_channel_distance: int = 5,  # <<< NEW PARAM: minimum distance between selected channels
    ):
        """
        Analyze DAS waveform data, compute SNR, coherence, and noise power for each trace,
        and select the best waveform in each clustered section based on combined scores,
        excluding outliers in SNR, RMS noise, and coherence before scoring.

        Parameters
        ----------
        n_final_select : int or None
            If set, selects exactly this many top traces by combined score 
            from the final set (instead of taking all passing the threshold).
        min_channel_distance : int
            Minimum number of channels between selected traces within the same section/subsection.
        """

        # ------------------------
        # Helper functions
        # ------------------------
        def print_progress_bar(iteration, total, prefix='', suffix='', length=40):
            percent = f"{100 * (iteration / float(total)):.1f}"
            filled_length = int(length * iteration // total)
            bar = '█' * filled_length + '-' * (length - filled_length)
            print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
            if iteration == total:
                print()

        def remove_outliers(data):
            q1 = np.percentile(data, 10)
            q3 = np.percentile(data, 90)
            return (data >= q1) & (data <= q3)

        def estimate_snr_profile(trace, sampling_rate, win_sec=0.5, step_sec=0.05):
            win_samples = int(win_sec * sampling_rate)
            step_samples = int(step_sec * sampling_rate)
            if len(trace) < win_samples:
                return np.array([])
            snr_vals = []
            for i in range(0, len(trace) - win_samples, step_samples):
                signal = trace[i:i + win_samples]
                noise = trace[:i] if i > 0 else np.zeros_like(signal)
                noise_power = np.mean(noise**2) if len(noise) > 0 else 1e-10
                signal_power = np.mean(signal**2)
                snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else -np.inf
                snr_vals.append(snr)
            return np.array(snr_vals)

        def detect_start_time(trace, sampling_rate, win_sec=2, step_sec=0.2,
                            smooth_sigma=2, skip_first_n=3, min_prominence=1.0):
            snr_profile = estimate_snr_profile(trace, sampling_rate, win_sec, step_sec)
            if len(snr_profile) == 0:
                return 0.0, np.array([])
            smoothed_snr = gaussian_filter1d(snr_profile, sigma=smooth_sigma)
            search_region = smoothed_snr[skip_first_n:]
            peaks, properties = find_peaks(search_region, prominence=min_prominence)
            start_idx = peaks[0] if len(peaks) > 0 else np.argmax(search_region)
            start_time = (start_idx + skip_first_n) * step_sec
            return start_time, smoothed_snr

        def compute_snr_rms(trace, start_signal=2, win=2, sampling_rate=100):
            start_sample = int((start_signal - win / 2) * sampling_rate)
            end_sample = int((start_signal + win / 2) * sampling_rate)
            if start_sample < 0 or end_sample > len(trace):
                return np.nan, np.nan
            noise = trace[:start_sample]
            sig = trace[start_sample:end_sample]
            if len(noise) == 0 or len(sig) == 0:
                return np.nan, np.nan
            noise_power = np.mean(noise**2)
            signal_power = np.mean(sig**2)
            if noise_power <= 0:
                return np.nan, np.nan
            snr_val = 10 * np.log10(signal_power / noise_power)
            rms_noise = 20 * np.log10(np.sqrt(noise_power))
            return snr_val, rms_noise

        def compute_coherence(did, das_ids, traces, sampling_rate, start_signal, win=2, max_lag=100):
            idx = np.where(das_ids == did)[0][0]
            half_gauge = gauge_length // 2
            neighbor_ids = das_ids[max(0, idx-half_gauge): min(len(das_ids), idx+half_gauge+1)]
            neighbor_ids = [nid for nid in neighbor_ids if nid != did]
            if not neighbor_ids:
                return 0.0
            start_idx = int((start_signal - win / 2) * sampling_rate)
            end_idx = int((start_signal + win / 2) * sampling_rate)
            ref = traces[idx][start_idx:end_idx]
            if len(ref) == 0:
                return 0.0
            coherence_scores = []
            for nid in neighbor_ids:
                nidx = np.where(das_ids == nid)[0][0]
                seg = traces[nidx][start_idx:end_idx]
                if len(seg) != len(ref):
                    continue
                ref_z = (ref - np.mean(ref)) / (np.std(ref) + 1e-10)
                seg_z = (seg - np.mean(seg)) / (np.std(seg) + 1e-10)
                corr = correlate(seg_z, ref_z, mode='full')
                mid = len(corr) // 2
                lag_corr = corr[mid-max_lag: mid+max_lag+1] / len(ref)
                coherence_scores.append(np.max(np.abs(lag_corr)))
            return np.mean(coherence_scores) if coherence_scores else 0.0

        def save_data_attributes_plot(data, title, filename_suffix):
            plt.figure()
            plt.title(title)
            plt.plot(data, '*')
            output_path = os.path.join(self.output_dir, f"{self.name}_{filename_suffix}.pdf")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

        def plot_metric(section_ids, values, title, ylabel, filename_suffix, color):
            plt.figure(figsize=(10, 4), dpi=300)
            plt.bar(section_ids, values, color=color)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.xticks(rotation=45)
            plt.grid(axis='y')
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, f"{self.name}_{filename_suffix}")
            plt.savefig(output_path)
            plt.close()

        def normalize(arr):
            ptp = np.ptp(arr)
            return (arr - np.min(arr)) / (ptp + 1e-10)

        # ------------------------
        # Main Processing
        # ------------------------
        traces = self.das_data
        das_ids = np.array(self.das_id)
        total_traces = len(traces)

        # 1. Start times
        if start_event is None:
            raw_start_times = []
            for i, (did, tr) in enumerate(zip(das_ids, traces), 1):
                start_t, _ = detect_start_time(tr, self.sampling_rate,
                                            win_sec=win_snr, step_sec=step_snr)
                raw_start_times.append(start_t)
                print_progress_bar(i, total_traces, prefix='Start detection:', suffix='Complete')
            smoothed = gaussian_filter1d(raw_start_times, sigma=2)
            start_signal = dict(zip(das_ids, smoothed))
        else:
            start_signal = {did: start_event for did in das_ids}

        # 2. SNR + RMS
        snr_dict, rms_dict = {}, {}
        for i, (did, tr) in enumerate(zip(das_ids, traces), 1):
            snr_val, rms_val = compute_snr_rms(tr, start_signal[did], win, self.sampling_rate)
            snr_dict[did], rms_dict[did] = snr_val, rms_val
            print_progress_bar(i, total_traces, prefix='SNR+RMS:', suffix='Complete')

        # 3. Coherence
        coherence_dict = {}
        for i, did in enumerate(das_ids, 1):
            coh = compute_coherence(did, das_ids, traces, self.sampling_rate, start_signal[did], win)
            coherence_dict[did] = coh
            print_progress_bar(i, total_traces, prefix='Coherence:', suffix='Complete')

        # 4. Outlier filtering + scores
        snrs = np.array([snr_dict[d] for d in das_ids])
        rms_noises = np.array([rms_dict[d] for d in das_ids])
        coherences = np.array([coherence_dict[d] for d in das_ids])

        snr_mask = remove_outliers(snrs)
        rms_mask = remove_outliers(rms_noises)
        coh_mask = remove_outliers(coherences)
        combined_mask = snr_mask & rms_mask & coh_mask

        snr_scores = np.zeros_like(snrs)
        rms_scores = np.zeros_like(rms_noises)
        coh_scores = np.zeros_like(coherences)
        snr_scores[combined_mask] = normalize(snrs[combined_mask])
        coh_scores[combined_mask] = normalize(coherences[combined_mask])
        rms_scores[combined_mask] = 1 - normalize(rms_noises[combined_mask])

        save_data_attributes_plot(snrs, 'SNR (dB)', 'snr')
        save_data_attributes_plot(coherences, 'Coherence', 'coherence')
        save_data_attributes_plot(rms_noises, 'RMS noise', 'rms')

        trace_scores = (snr_score_weight * snr_scores +
                        coherence_score_weight * coh_scores +
                        noise_rms_weight * rms_scores)

        positive_scores = trace_scores[trace_scores > 0]
        lower_bound = np.percentile(positive_scores, percentile) if len(positive_scores) > 0 else 0

        # 5. Group by sections (cluster labels)
        labels = self.cluster_labels[: len(das_ids)]
        section_data = {}
        for did, lb in zip(das_ids, labels):
            if lb < 0:
                continue
            section_data.setdefault(lb, []).append(
                (did, snr_scores[np.where(das_ids==did)[0][0]],
                coh_scores[np.where(das_ids==did)[0][0]],
                rms_scores[np.where(das_ids==did)[0][0]],
                trace_scores[np.where(das_ids==did)[0][0]])
            )

        section_scores = {}
        section_keys = list(section_data.keys())
        selected_channels_global = []  # tracks all selected channel indices across sections

        for i, lb in enumerate(section_keys, 1):
            arr = sorted(section_data[lb], key=lambda x: x[0])
            chunks = [arr[i:i + subsection_size] for i in range(0, len(arr), subsection_size)]
            for c_i, sec in enumerate(chunks):
                sec_id = f"{lb}.{c_i}" if len(chunks) > 1 else str(lb)
                if not sec:
                    continue
                avg_section_score = np.mean([x[4] for x in sec])
                if avg_section_score < lower_bound:
                    continue
                
                # --- Select best trace respecting min_channel_distance globally ---
                sec_sorted = sorted(sec, key=lambda x: x[4], reverse=True)
                selected_in_sec = []
                selected_channels_local = []

                for entry in sec_sorted:
                    did = entry[0]
                    idx = np.where(das_ids == did)[0][0]

                    # Check distance to local subsection and global previous selections
                    if all(abs(idx - prev_idx) >= min_channel_distance for prev_idx in selected_channels_local + selected_channels_global):
                        selected_in_sec.append(entry)
                        selected_channels_local.append(idx)
                        selected_channels_global.append(idx)  # add to global
                        break  # select only one per subsection

                if selected_in_sec:
                    section_scores[sec_id] = selected_in_sec[0]

            print_progress_bar(i, len(section_keys), prefix='Section select:', suffix='Complete')


        if not section_scores:
            print("No sections passed threshold.")
            return [], [], [], []

        # --- Option to select only top-N final traces ---
        if n_final_select is not None:
            # sort by score (descending) first
            sorted_by_score = sorted(
                section_scores.items(),
                key=lambda kv: kv[1][4],
                reverse=True
            )
            # take top N
            top_n = dict(sorted_by_score[:n_final_select])
            # then reorder by section id (numeric order, not score)
            def parse_section_id(sid):
                parts = sid.split(".")
                return tuple(map(int, parts))  # supports multi-level ids like "2.1"
            section_scores = dict(sorted(top_n.items(), key=lambda kv: parse_section_id(kv[0])))


        selected_ids, selected_snr, selected_coh, selected_rms, selected_scores = zip(*section_scores.values())

        # Save results
        selected_traces_dir = os.path.join(self.output_dir, "selected_traces")
        os.makedirs(selected_traces_dir, exist_ok=True)
        outpath = os.path.join(selected_traces_dir, "selected_waveforms.txt")
        with open(outpath, "w") as f:
            f.write("section_id\tdas_id\tsnr\tcoherence\trms\tcombined\n")
            for sid, vals in section_scores.items():
                did, snr_val, coh_val, rms_val, score_val = vals
                f.write(f"{sid}\t{did}\t{snr_val:.2f}\t{coh_val:.3f}\t{rms_val:.3f}\t{score_val:.3f}\n")

        plot_metric(list(section_scores.keys()), selected_snr, 'Best Section SNR', 'SNR', 'best_snr.pdf', 'tab:green')
        plot_metric(list(section_scores.keys()), selected_coh, 'Best Section Coherence', 'Coherence', 'best_coh.pdf', 'tab:orange')
        plot_metric(list(section_scores.keys()), selected_rms, 'Best Section RMS noise', 'RMS', 'best_rms.pdf', 'tab:red')
        plot_metric(list(section_scores.keys()), selected_scores, 'Best Section Combined', 'Score', 'best_combined.pdf', 'tab:purple')

        return list(selected_ids), list(selected_snr), list(selected_coh), list(selected_rms)



    def save_selected_traces_to_mseed(self, save_non_selected=False, n_subset_orion=None):
        """
        Saves:
        - full_selected_traces.mseed: stacked traces from full Orion selection
        - selected_traces.mseed: stacked traces from subsampled Orion selection
        - full_uniform_traces.mseed: unstacked traces from non-selected channels (same count as full selection)
        - non_selected.mseed (optional): unstacked every 10th non-selected trace
        """
        output_path = os.path.join(self.output_dir, "selected_traces")
        os.makedirs(output_path, exist_ok=True)
        selection_path = os.path.join(output_path, "selected_waveforms.txt")

        # --- Read full original Orion selection ---
        full_selected_traces = []
        with open(selection_path, "r") as file:
            next(file)  # Skip header
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                section_id = parts[0]
                das_id = int(float(parts[1]))
                full_selected_traces.append((section_id, das_id))

        # --- Subsample Orion selection if requested ---
        selected_traces = full_selected_traces
        if n_subset_orion and n_subset_orion < len(full_selected_traces):
            step = len(full_selected_traces) / n_subset_orion
            selected_traces = [full_selected_traces[int(i * step)] for i in range(n_subset_orion)]

        num_channels = self.das_data.shape[0]

        # --- Helper: stack ±4 before, +5 after neighbors ---
        def create_stacked_stream(traces_list):
            stacked_stream = Stream()
            for _, das_id in traces_list:
                neighbors = []
                for offset in range(-4, 6):  # -4 to +5
                    ni = das_id + offset
                    if 0 <= ni < num_channels:
                        neighbors.append(self.das_data[ni])
                if neighbors:
                    stacked_data = np.sum(neighbors, axis=0)
                    max_abs_value = np.max(np.abs(stacked_data))
                    if max_abs_value > 0:
                        stacked_data = stacked_data / max_abs_value
                    stacked_trace = Trace(data=stacked_data)
                    stacked_trace.stats.network = "DAS"
                    stacked_trace.stats.station = str(das_id)
                    stacked_trace.stats.channel = "HHE"
                    stacked_trace.stats.starttime = UTCDateTime(0)
                    stacked_trace.stats.sampling_rate = self.sampling_rate
                    stacked_stream.append(stacked_trace)
            return stacked_stream

        # --- Save stacked full Orion selection ---
        full_stacked_stream = create_stacked_stream(full_selected_traces)
        full_stacked_file = os.path.join(output_path, "full_selected_traces.mseed")
        full_stacked_stream.write(full_stacked_file, format="MSEED")

        # --- Save stacked final selection (subsampled) ---
        final_stacked_stream = create_stacked_stream(selected_traces)
        final_stacked_file = os.path.join(output_path, "selected_traces.mseed")
        final_stacked_stream.write(final_stacked_file, format="MSEED")

        # --- Save unstacked uniformly spaced traces from non-selected channels ---
        selected_ids = {das_id for _, das_id in full_selected_traces}
        all_ids = np.array([i for i in range(num_channels) if i not in selected_ids])

        uniform_count = len(full_selected_traces)
        if len(all_ids) < uniform_count:
            raise ValueError(f"Not enough non-selected channels ({len(all_ids)}) to create {uniform_count} uniform traces.")

        # Uniformly select IDs from the non-selected pool
        uniform_ids = np.linspace(0, len(all_ids) - 1, uniform_count, dtype=int)
        uniform_selected_ids = all_ids[uniform_ids]

        full_uniform_stream = Stream()
        for das_id in uniform_selected_ids:
            data = self.das_data[das_id]
            trace = Trace(data=data)
            trace.stats.network = "DAS"
            trace.stats.station = str(das_id)
            trace.stats.channel = "HHE"
            trace.stats.starttime = UTCDateTime(0)
            trace.stats.sampling_rate = self.sampling_rate
            full_uniform_stream.append(trace)

        full_uniform_file = os.path.join(output_path, "full_uniform_traces.mseed")
        full_uniform_stream.write(full_uniform_file, format="MSEED")

        # --- Optionally save every 10th non-selected trace ---
        if save_non_selected:
            all_non_selected_stream = Stream()
            every_10th_ids = [i for i in range(0, num_channels, 10) if i not in selected_ids]
            for das_id in every_10th_ids:
                data = self.das_data[das_id]
                trace = Trace(data=data)
                trace.stats.network = "DAS"
                trace.stats.station = str(das_id)
                trace.stats.channel = "HHE"
                trace.stats.starttime = UTCDateTime(0)
                trace.stats.sampling_rate = self.sampling_rate
                all_non_selected_stream.append(trace)

            non_selected_output_file = os.path.join(output_path, "non_selected.mseed")
            all_non_selected_stream.write(non_selected_output_file, format="MSEED")

        # --- Summary ---
        print(
            f"Saved full selected (stacked) traces: {len(full_stacked_stream)} traces\n"
            f"Saved final selected (stacked) traces: {len(final_stacked_stream)} traces\n"
            f"Saved full uniform (unstacked) traces: {len(full_uniform_stream)} traces"
            f"{', and ' + str(len(all_non_selected_stream)) + ' non-selected traces' if save_non_selected else ''} "
            f"to '{output_path}'"
        )


    def load_selected_traces_from_mseed_and_plot(self):
        input_dir: str = os.path.join(self.output_dir, "selected_traces")
        trace_dict = {}
        stacked_dict = {}
        picks_dict = {}

        selected_file = os.path.join(input_dir, "full_selected_traces.mseed")
        stacked_file = os.path.join(input_dir, "full_selected_traces.mseed")
        picks_file = os.path.join(input_dir, "selected_waveforms.txt")

        # --- Load picks from .txt ---
        if os.path.exists(picks_file):
            with open(picks_file, "r") as f:
                lines = f.readlines()[1:]  # Skip header
                for line in lines:
                    parts = line.strip().split("\t")
                    if len(parts) < 9:
                        continue
                    section_id = parts[0]
                    p_pick_str = parts[7]
                    s_pick_str = parts[8]
                    try:
                        p_pick = UTCDateTime(p_pick_str) if p_pick_str != "None" else None
                        s_pick = UTCDateTime(s_pick_str) if s_pick_str != "None" else None
                    except Exception:
                        p_pick = s_pick = None
                    picks_dict[section_id] = (p_pick, s_pick)

        # --- Load selected traces ---
        try:
            selected_stream = read(selected_file)
            for tr in selected_stream:
                section_id = tr.stats.station  # stored as section_id
                trace_dict[section_id] = tr
        except Exception as e:
            print(f"Error reading selected traces: {e}")

        # --- Load stacked traces ---
        try:
            stacked_stream = read(stacked_file)
            for tr in stacked_stream:
                section_id = tr.stats.station  # stored as section_id
                stacked_dict[section_id] = tr
        except Exception as e:
            print(f"Error reading stacked traces: {e}")

        if not trace_dict or not stacked_dict:
            print("No traces found.")
            return {}

        # --- Sort section_ids numerically ---
        def sort_key(k):
            try:
                return float(k)
            except ValueError:
                return float('inf')

        sorted_section_ids = sorted(trace_dict.keys(), key=sort_key)

        # --- Plot ---
        cmap = cm.get_cmap('tab20c')
        plt.figure(figsize=(12, 6 + len(trace_dict) * 0.2), dpi=300)

        for i, section_id in enumerate(sorted_section_ids):
            trace = trace_dict.get(section_id)
            stacked_trace = stacked_dict.get(section_id)
            if trace is None or stacked_trace is None:
                continue

            sr = trace.stats.sampling_rate
            npts = trace.stats.npts
            time = np.linspace(0, npts / sr, npts)
            offset = i * 2

            # Normalize original trace
            data_orig = trace.data.astype(np.float32)
            data_orig /= np.max(np.abs(data_orig)) + 1e-12

            # Normalize stacked trace
            data_stack = stacked_trace.data.astype(np.float32)
            data_stack /= np.max(np.abs(data_stack)) + 1e-12

            plt.plot(time, data_orig + offset, color="lightgray", linewidth=0.8)
            plt.plot(time, data_stack + offset, color="black", linewidth=1.2)

            # Plot P and S picks
            p_pick, s_pick = picks_dict.get(section_id, (None, None))
            if p_pick:
                p_time = (p_pick - trace.stats.starttime)
                plt.plot(p_time, offset, marker="*", color="blue", markersize=10, label="P Pick" if i == 0 else "")
            if s_pick:
                s_time = (s_pick - trace.stats.starttime)
                plt.plot(s_time, offset, marker="*", color="red", markersize=10, label="S Pick" if i == 0 else "")

        plt.xlabel("Time [s]")
        plt.ylabel("Section ID")
        plt.yticks([i * 2 for i in range(len(sorted_section_ids))], sorted_section_ids)
        plt.title(f"Stacked vs. Individual DAS Traces with Picks – {self.name}")
        plt.grid(True)

        # Only show pick legend once
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            plt.legend(by_label.values(), by_label.keys(), loc="upper right")

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f"{self.name}_stack_vs_individual_sta_lta_plot.pdf")
        plt.savefig(output_path)
        plt.close()

        print(f"Plot saved to: {output_path}")
        print(f"Processed and plotted {len(stacked_dict)} stacked traces from '{self.output_dir}'")

        return trace_dict

    def plot_selected_traces_on_data(self):
        """Generate two figures: one with normalized DAS data and one overlayed with selected traces."""

        output_path = os.path.join(self.output_dir, "selected_traces")
        wav = os.path.join(output_path, "selected_waveforms.txt")

        plt.figure(figsize=(10, 10), dpi=200)

        num_traces, num_samples = self.das_data.shape
        duration_sec = num_samples / self.sampling_rate

        # Time axis
        time_vector = np.linspace(0, duration_sec, num_samples)

        # Show all traces as an image (background)
        extent = [time_vector[0], time_vector[-1], num_traces, 0]
        data = self.das_data

        # Normalize rows
        row_max = np.max(np.abs(data), axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        normalized_data = data / row_max

        im = plt.imshow(
            normalized_data,
            aspect="auto",
            cmap="coolwarm",
            extent=extent,
            vmin=-0.2,
            vmax=0.2,
            alpha=0.7
        )
        plt.colorbar(im, label="Amplitude")

        plt.xlabel("Relative Time [s]")
        plt.ylabel("DAS ID")
        plt.title(f"Automatic DAS channel selection: {self.name}")

        plt.legend(loc="upper right")

        output_path = os.path.join(self.output_dir, f"{self.name}_das_data.pdf")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Plot saved to: {output_path}")

        # --- Second figure with selected traces ---
        plt.figure(figsize=(7, 6), dpi=200)

        im = plt.imshow(
            normalized_data,
            aspect="auto",
            cmap="coolwarm",
            extent=extent,
            vmin=-0.2,
            vmax=0.2,
            alpha=0.7
        )
        plt.colorbar(im, label="Amplitude", shrink=0.5, pad=0.1)

        plt.xlabel("Relative Time [s]")
        plt.ylabel("DAS ID")
        plt.title("Automatic DAS channel selection")

        # --- Read selected traces from file ---
        selected_traces = []
        with open(wav, "r") as f:
            for line in f:
                if line.strip() == "":
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    section_id = parts[0]
                    try:
                        trace_index = int(float(parts[1]))
                        selected_traces.append((section_id, trace_index))
                    except ValueError:
                        print(f"Skipping invalid line: {line.strip()}")

        unique_labels = sorted(set(self.cluster_labels))
        unique_non_noise = [lbl for lbl in unique_labels if lbl != -1]
        num_clusters = len(unique_non_noise)

        base_palette = sns.color_palette("tab20c", 20)

        def transform_color(color, cycle):
            arr = np.array(color)
            if cycle == 0:
                return color
            elif cycle == 1:
                return sns.desaturate(color, 0.5)
            elif cycle == 2:
                return tuple(np.clip(arr * 0.6, 0, 1))
            elif cycle == 3:
                return tuple(0.3 + 0.7 * arr)
            elif cycle == 4:
                return tuple(np.clip(arr * 1.2, 0, 1))
            else:
                return transform_color(color, cycle % 5)

        extended_palette = []
        for i in range(num_clusters):
            base_color = base_palette[i % 20]
            cycle = i // 20
            extended_palette.append(transform_color(base_color, cycle))

        cmap = ListedColormap(extended_palette)
        norm = BoundaryNorm(unique_non_noise + [max(unique_non_noise) + 1], cmap.N)

        # --- Overlay selected traces with cluster colors ---
        vertical_scale = 50
        for section_id, trace_index in selected_traces:
            if not (0 <= trace_index < num_traces):
                print(f"Skipping trace_index {trace_index}: out of DAS data range ({num_traces})")
                continue
            if not (0 <= trace_index < len(self.cluster_labels)):
                print(f"Skipping trace_index {trace_index}: out of cluster_labels range ({len(self.cluster_labels)})")
                continue

            tr = self.das_data[trace_index, :]
            if np.max(np.abs(tr)) == 0:
                continue
            tr = tr / np.max(np.abs(tr))

            # find cluster id for this trace
            cluster_id = self.cluster_labels[trace_index]

            if cluster_id == -1:
                color = "black"  # noise
            else:
                color = cmap(norm(cluster_id))

            plt.plot(
                time_vector,
                tr * vertical_scale + trace_index,
                color="black",
                linewidth=1.1,
                alpha=0.8,
                zorder=9,
            )  # outline
            plt.plot(
                time_vector,
                tr * vertical_scale + trace_index,
                color=color,
                linewidth=1,
                alpha=0.5,
                zorder=10,
                label = 'Selected channel'
            )  # colored trace

        # --- Build legend with unique cluster IDs ---
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc="upper right")

        # Save figure
        output_path = os.path.join(
            self.output_dir, f"{self.name}_automatic_channel_selection_with_selected_traces.pdf"
        )
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Plot saved to: {output_path}")


    def save_clustering(self, filename):
        
        to_save = {
            "cluster_labels": self.cluster_labels,
            "scaler": self.scaler,
            "chunk_centers": getattr(self, "chunk_centers", None),
            # add other attributes if needed
        }
        joblib.dump(to_save, filename)
        print(f"Clustering results saved to {filename}")

    def load_clustering(self, filename):
        
        data = joblib.load(filename)
        self.cluster_labels = data["cluster_labels"]
        self.scaler = data["scaler"]
        self.chunk_centers = data.get("chunk_centers", None)
        print(f"Clustering results loaded from {filename}")

