# Knee Adduction and Knee Flexion Moment estimation during walking via a Wearable Inertial Sensor Network

## Prerequisite
a. install numpy, pandas, scipy-kit, pytorch, 

The working flow of this code repository:

1. generate_combined_data.py: This will iterate, preprocess, and supplement all raw data and generate a combined, cropped, synchronized csv file. The file will contain time-feature data and one of the features indicates the step id.
2. generate_step_data.py: This will iterate, filter, and supplement all the combined csv file generated from `generate_combined_data.py` and assembly into an hdf5 file. The assembly hdf5 file will contain step-time-feature data.
3. tian_model.py: hmmmmmmmm, Alan, can you explain it?
