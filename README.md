### Dependencies (Programming Languages and Software)
1. MATLAB R2016B
2. Python 3.6.0
3. ImageJ Version: 2.0.0-rc-69
4. Python dependences located in python_dependencies.txt

#### Operating System Tested On: macOS Sierra V 10.12.6

#### File Descriptions
1. create_metadata_file.ijm: ImageJ script to create a csv (as shown in example_data/external_data)
2. create_track_matrices.mlx: Matlab script for taking data obtained from Utrack (software created by the Danuser Lab at UTSW) to csv format compatible with our scripts
3. extract_features_from_track_matrices.ipynb: python script which takes csv format data as imput (as shown in example_data/track_matrices) and outputs multiple dataframes stored in python pickle objects (example_data/pickle_objs) 

### How to use: 
To generate track data run extract_features_from_track_matrices.ipynb on jupyter notebook.
To visualize the track matrices run load_track_data.ipynb
The expected output is a pandas dataframe consisting of MT tracks x features. Extracting features may take as long as 15 minutes depending on your system.

### Installation:
Make sure all dependencies are fulfilled and download all source files to local computer. Installing dependencies may take some time (~10 minutes for me)
