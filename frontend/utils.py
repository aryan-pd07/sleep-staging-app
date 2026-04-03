import numpy as np
import pandas as pd
import mne
import tempfile
import os

# Standardizing Label Mapping (String -> Int)
# Maps various common annotation formats to your model's 0-4 scale
LABEL_MAP = {
    # Standard AASM
    "sleep stage w": 0, "wake": 0,
    "sleep stage 1": 1, "n1": 1, "sleep stage n1": 1,
    "sleep stage 2": 2, "n2": 2, "sleep stage n2": 2,
    "sleep stage 3": 3, "n3": 3, "sleep stage n3": 3,
    "sleep stage 4": 3, # Merge N4 into N3
    "sleep stage r": 4, "rem": 4, "sleep stage rem": 4,
    "movement": 0, "unknown": 0 # Default artifacts to Wake or Ignore
}

def process_uploaded_file(uploaded_file):
    """
    Handles loading raw EDF/CSV data.
    """
    fname = uploaded_file.name.lower()
    
    if fname.endswith('.edf'):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tfile:
                tfile.write(uploaded_file.getvalue())
                tfile_path = tfile.name
            
            raw = mne.io.read_raw_edf(tfile_path, preload=True, verbose=False)
            os.unlink(tfile_path)
            
            if raw.info['sfreq'] != 100:
                raw.resample(100, npad="auto")
            
            return None, raw, None
        except Exception as e:
            return None, None, f"EDF Error: {str(e)}"

    # Handle Simple Files (CSV/NPY)
    try:
        if fname.endswith('.npy'):
            data = np.load(uploaded_file)
        elif fname.endswith('.csv'):
            df = pd.read_csv(uploaded_file, header=None)
            data = df.values.flatten()
        else:
            return None, None, "Unsupported file format."
        
        data = data.astype('float32')
        return data, None, None
    except Exception as e:
        return None, None, f"Error: {str(e)}"

def load_ground_truth(file_obj, num_epochs, start_epoch_idx):
    """
    Parses a Hypnogram file (EDF/TXT) and extracts labels for the specific 
    time range currently being analyzed.
    """
    try:
        fname = file_obj.name.lower()
        labels = []

        # CASE A: EDF+ Annotations (Standard for PhysioNet/medical data)
        if fname.endswith('.edf'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tfile:
                tfile.write(file_obj.getvalue())
                tpath = tfile.name
            
            # Read annotations
            annot = mne.read_annotations(tpath)
            os.unlink(tpath)

            # Convert annotations to per-epoch labels (30s)
            # We iterate through the requested epochs and check the annotation at that time
            for i in range(num_epochs):
                current_time = (start_epoch_idx + i) * 30
                
                # Find annotation active at this timestamp
                # (Simple logic: take the label that covers the midpoint of the epoch)
                midpoint = current_time + 15 
                
                found_label = "unknown"
                for desc, onset, duration in zip(annot.description, annot.onset, annot.duration):
                    if onset <= midpoint < (onset + duration):
                        found_label = desc.lower()
                        break
                
                # Map to Int
                label_int = LABEL_MAP.get(found_label, -1) # -1 if not found
                labels.append(label_int)
                
            return np.array(labels)

        # CASE B: TXT/CSV (Line-by-line stages)
        elif fname.endswith('.txt') or fname.endswith('.csv'):
            df = pd.read_csv(file_obj, header=None)
            full_labels = df.iloc[:, 0].astype(str).str.lower().map(LABEL_MAP).values
            
            # Slice specific range
            end_idx = start_epoch_idx + num_epochs
            if end_idx > len(full_labels):
                return None # Dimension mismatch
            
            return full_labels[start_epoch_idx : end_idx]

    except Exception as e:
        print(f"GT Error: {e}")
        return None

def get_dummy_data():
    # 30 mins data
    return np.random.normal(0, 1, (180000,)), np.random.normal(0, 1, (180000,))
