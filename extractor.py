import numpy as np
from tdt import read_block

def extract_fiber_photometry_data(tdt_block_path, signal_channel_name, control_channel_name, pct_channel_name):
    """
    Extracts signal, control, and PCT channel data from a TDT block.

    Args:
        tdt_block_path (str): Path to the TDT block folder.
        signal_channel_name (str): Name of the signal channel (e.g., '465A').
        control_channel_name (str): Name of the control channel (e.g., '415A').
        pct_channel_name (str): Name of the PCT channel (e.g., 'PCT1').

    Returns:
        dict: {
            'signal': np.ndarray,
            'control': np.ndarray,
            'pct': np.ndarray,
            'sr': float,
            'time': np.ndarray
        }
    """
    # Load the block
    print(f"Loading TDT block from: {tdt_block_path}")
    
    block = read_block(tdt_block_path)

    print(block.streams.keys())
    print(block.epocs.keys())
    # Extract data streams
    try:
        signal = block.streams[signal_channel_name].data.flatten()
        control = block.streams[control_channel_name].data.flatten()
    #    pct = block.epocs[pct_channel_name].onset
    except KeyError as e:
        raise KeyError(f"Channel name not found in block streams: {e}")

    # Sampling rate (assumed same for all streams)
    sr = block.streams[signal_channel_name].fs

    # Create timestamps vector
    n_samples = len(signal)
    time = np.arange(n_samples) / sr

    #print(f"Extracted data lengths: Signal={len(signal)}, Control={len(control)}, PCT={len(pct)}")
    print(f"Sampling rate: {sr} Hz")
    print(f"Duration: {time[-1]:.2f} seconds")
    #print(f"First 5 trial start times (PCT epocs): {pct[:5] if pct is not None else 'No PCT data'}")

    return {
        'signal': signal,
        'control': control,
     #   'pct': pct,
        'sr': sr,
        'time': time
    }

if __name__ == "__main__":
    # User input here:
    tdt_block_path = r"C:\Users\ParekhLab\Downloads\18_2dLightSucroseTest\18_2dLightSucroseTest"  # Change this to your actual path
    signal_channel_name = '_465A'   # Change if different in your data
    control_channel_name = '_415A'  # Change if different in your data
    pct_channel_name = 'PtC1'      # Change if different in your data

    data = extract_fiber_photometry_data(tdt_block_path, signal_channel_name, control_channel_name, pct_channel_name)

    # Optional: save to npz for later use
    np.savez('fiber_photometry_data5.npz', 
             signal=data['signal'], 
             control=data['control'], 
             #pct=data['pct'], 
             sr=data['sr'], 
             time=data['time'])
    print("Data saved to fiber_photometry_data.npz")
