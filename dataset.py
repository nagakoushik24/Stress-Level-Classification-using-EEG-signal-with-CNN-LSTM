import os
import numpy as np
import pandas as pd
import scipy
import variables as v


def load_dataset(data_type="ica_filtered", test_type="Arithmetic"):
    '''
    Loads data from the SAM 40 Dataset.

    Args:
        data_type (string): The data type to load. Defaults to "ica_filtered".
        test_type (string): The test type to load. Defaults to "Arithmetic".

    Returns:
        ndarray: The specified dataset.

    '''
    assert (test_type in v.TEST_TYPES)

    assert (data_type in v.DATA_TYPES)

    if data_type == "raw":
        dir = v.DIR_RAW
        data_key = 'Data'
    elif data_type == "wt_filtered":
        dir = v.DIR_FILTERED
        data_key = 'Clean_data'
    else:
        dir = v.DIR_ICA_FILTERED
        data_key = 'Clean_data'

    dataset = np.empty((120, 32, 3200))

    counter = 0
    for filename in os.listdir(dir):
        if test_type not in filename:
            continue

        f = os.path.join(dir, filename)
        data = scipy.io.loadmat(f)[data_key]
        dataset[counter] = data
        counter += 1
    return dataset


def load_labels():
    '''
    Loads labels from the dataset without transforming them to binary values,
    keeping the original 1-10 range.

    Returns:
        ndarray: The labels with values in the 1-10 range.
    '''
    labels = pd.read_excel(v.LABELS_PATH)
    labels = labels.rename(columns=v.COLUMNS_TO_RENAME)
    labels = labels[1:]  # Adjust if necessary
    labels = labels.astype("int")  # Ensure they are integers

    return labels


def format_labels(labels, test_type="Arithmetic", epochs=1):
    '''
    Filter the labels and repeat for the specified amount of epochs.

    Args:
        labels (ndarray): The labels.
        test_type (string): The test_type to filter by. Defaults to "Arithmetic".
        epochs (int): The amount of epochs. Defaults to 1.

    Returns:
        ndarray: The formatted labels with values in the 1-10 range.
    '''
    assert (test_type in v.TEST_TYPES)

    formatted_labels = []
    for trial in v.TEST_TYPE_COLUMNS[test_type]:
        if trial in labels.columns:
            formatted_labels.append(labels[trial])
        else:
            print(f"Warning: {trial} not found in labels")

    formatted_labels = pd.concat(formatted_labels).to_numpy()
    formatted_labels = formatted_labels.repeat(epochs)

    return formatted_labels


def split_data(data, sfreq):
    '''
    Splits EEG data into epochs with length 1 sec.

    Args:
        data (ndarray): EEG data.
        sfreq (int): The sampling frequency.

    Returns:
        ndarray: The epoched data.

    '''

    n_trials, n_channels, n_samples = data.shape

    epoched_data = np.empty((n_trials, n_samples // sfreq, n_channels, sfreq))
    for i in range(data.shape[0]):
        for j in range(data.shape[2] // sfreq):
            epoched_data[i, j] = data[i, :, j * sfreq:(j + 1) * sfreq]
    return epoched_data
