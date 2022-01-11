# TinySleepNet-pytorch
A simple implementation of TinySleepNet with some improvements.

## Environment

* pytorch >=1.6.0
* scikit-learn

## Prepare dataset ##
Download Sleep-EDF dataset(https://physionet.org/pn4/sleep-edfx/),  and save in `\data`.

Then run the following script to extract specified EEG channels and their corresponding sleep stages. All the following steps are operated on SC subjects. 

    `python read_sleepedf.py --data_dir data/sleepedf/sleep-cassette --output_dir data/eeg_fpz_cz --select_ch 'EEG Fpz-Cz'`
    `python read_sleepedf.py --data_dir data/sleepedf/sleep-cassette --output_dir data/eeg_pz_oz --select_ch 'EEG Pz-Oz'`

    `python prepare_sleepedf.py`

## Run
 Run python file progress.py

You can set all of the parameters in function main.
