# run_channel_selection.py
import os
from orionidas import DASChannelSelector  # replace with the actual module name
from tqdm import tqdm  # add this import at the top
import glob

#fiber geometry path and simplified geometry path 

geometry_csv_path = '/home/emanuele/post_doc/ORION/safe/synth_test/synth_das_coordinates.txt'
simplified_geometry_csv = '/home/emanuele/post_doc/ORION/safe/synth_test/simplified_geometry_synth.csv'

#directory containing .mseed and .h5 files


#das_directory = '/home/emanuele/post_doc/codici/kefalonia/input_detected_events'
das_directory = '/home/emanuele/post_doc/ORION/safe/synth_test'

# how names of the files are organized
mseed_pattern = os.path.join(das_directory, 'noa2024*_das.mseed')
h5_pattern = os.path.join(das_directory, '7M_S_*.h5')   # for kef0365.h5-style files
npy_pattern = os.path.join(das_directory, 'synth_salvus_noise.npy')      # for generic .npy files
#npy_pattern = os.path.join(das_directory, 'synth_salvus.npy')      # for generic .npy files


# Matching files
mseed_files = sorted(glob.glob(mseed_pattern))
h5_files = sorted(glob.glob(h5_pattern))
npy_files = sorted(glob.glob(npy_pattern))

# Build name/path pairs for .mseed files
mseed_names_and_paths = [
    (
        os.path.basename(f)
        .replace('noa2024', '')
        .replace('_das.mseed', ''),
        f
    )
    for f in mseed_files
]

# Build name/path pairs for .h5 files (strip .h5)
h5_names_and_paths = [
    (
        os.path.splitext(os.path.basename(f))[0],  # removes .h5
        f
    )
    for f in h5_files
]

# Build name/path pairs for .npy files (strip .npy)
npy_names_and_paths = [
    (
        os.path.splitext(os.path.basename(f))[0],  # removes .npy
        f
    )
    for f in npy_files
]

# Combine all
names_and_paths = mseed_names_and_paths + h5_names_and_paths + npy_names_and_paths



#path to store the clustering resust to avoid repeating the spatial clustering step
clustering_save_path = '/home/emanuele/post_doc/ORION/safe/clustering_results_synth.pkl'


def main():
    output_root = '/home/emanuele/post_doc/ORION/safe/output_orion_synth_noise'
    #output_root = '/home/emanuele/post_doc/ORION/safe/output_orion_synth'

    for i, (name, das_data_path) in enumerate(tqdm(names_and_paths, desc="Processing datasets")):

        print(f"\nProcessing: {name}")

        output_dir = os.path.join(output_root, name)

        selector = DASChannelSelector(
            name=name,
            geometry_path=geometry_csv_path,
            das_data_path=das_data_path,
            output_dir=output_dir  # Pass output directory here
        )
        
        #first geometrical clusterization step (done only once)

        if i == 0:
            steps = [
                ("Simplify geometry CSV", lambda: selector.simplify_geometry_csv(simplified_geometry_csv)),
                ("Load geometry", lambda: selector.load_geometry(simplified_geometry_csv)),
                ("Plot geometry", selector.plot_geometry),
                ("Compute average azimuth", lambda: selector.compute_average_azimuth(chunk_size=10)), #chunk size in number of channels to average the azimuth 
                ("Estimate DBSCAN eps", lambda: setattr(
                    selector, 'eps_est', selector.estimate_dbscan_eps(min_samples=10)
                )),  # minimum number of samples to define a family

                ("Cluster by azimuth distance", lambda: selector.cluster_by_azimuth_distance(
                    eps=selector.eps_est,   # use estimated eps instead of fixed 0.5
                    min_samples=10,
                    window_size=10
                )),
                ("Plot geometry with clusters", selector.plot_geometry_with_clusters),
                ("Save clustering", lambda: selector.save_clustering(clustering_save_path)),
            ]


            for desc, func in tqdm(steps, desc="Spatial clustering steps"):
                func()

        #if not the first iteration, load the clustering results from the first run

        else:


            selector.load_clustering(clustering_save_path)

            if not hasattr(selector, 'chunk_centers') or selector.chunk_centers is None:
                selector.compute_chunk_centers()


        selector.read_das_data(
            sampling_rate=400, 
            detrend_data=True,
            apply_filter=True,
            apply_taper=True,
            taper_alpha=0.01,
            low_freq=1,
            high_freq=100,
            dx=2, 
            remove_k0=False
        )

        selector.analyze_waveforms_and_select_best(
            gauge_length=10,    
            subsection_size=20,   # minimum number of channels to define a subsection
            win_snr=0.4,            
            step_snr=0.2, 
            win=2,               # for coherence 
            start_event=None,
            percentile=10, 
            sta_window_sec=0.5,   # shorter STA window
            lta_window_sec=5.0,   # slightly shorter LTA
            sta_lta_on=3.5,       # a bit lower threshold for ON
            sta_lta_off=1.2,      # a bit lower threshold for OFF          
            snr_score_weight=1,
            coherence_score_weight=1,
            noise_rms_weight=0.3,
            n_final_select=10,     # <<< NEW: select top 10 traces only
            min_channel_distance= 10
        )

        

        selector.save_selected_traces_to_mseed(save_non_selected=False,
    n_subset_orion=10
)
        selector.load_selected_traces_from_mseed_and_plot()
        selector.plot_selected_traces_on_data()

if __name__ == "__main__":
    main()
