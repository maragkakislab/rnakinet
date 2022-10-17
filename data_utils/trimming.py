def primer_trim(signal, window_size=200, threshold=1.9, min_elements=25):

    min_trim = 10
    signal = signal[min_trim:]
    num_windows = len(signal) // window_size

    seen_peak = False
    for pos in range(num_windows):
        start = pos * window_size
        end = start + window_size
        window = signal[start:end]
        if len(window[window > threshold]) > min_elements or seen_peak:
            seen_peak = True
            if window[-1] > threshold:
                continue
            return min(end + min_trim, len(signal))
    return min_trim