# ML_NN_Project: Multivariate across-subject decoding of MEG evoked responses from repeated visual food cues

### Project Description and Goals:
The codes contained in **src** implement:
1. General binary and multiclass decoding for exploring the decodability of various tasks across various windows and sample-feature combinations.
2. Across time decoding using SlidingEstimator and GeneralizingEstimator for finding onset and duration of food vs. neutral decodability, and uncovering the neural activation pattern. 
3. Representational similarity analysis using pairwise decoding of the 18 original conditions for unveiling the linearity of separability of food and non-food
4. Weight projection using Haufe transformation for identifying the most contributing sensors to food vs. non-food decodability.

### Data Description:
Each participant from the 42 participants in the study was shown image repetitions. Subject numbers are important as these are used as basis for making subdirectories.
There were 18 conditions in the experiment, the images were categorized into 3 main semantic categories (food, positive, and neutral), each image was shown twice (1st and 2nd presentation), and in varying lags between the repetitions (short, medium, and long).
The participants' brain activity was recorded using a MEG 4D-Neuroimaging system. 
The raw data was preprocessed using Matlab, a low band pass filter of 30 Hz and a high band pass filter of 1 Hz were used,and  artifacts of blinks and heart beats were excluded using ICA. The data was epoched in the time range of -0.3-0.8 s relative to event/stimuli onset. 

The data used in the implementation is:
* Epoched data in a fif format.
* A raw MEG 4D recording for obtaining sensor locations. 
* config and hsfile (head shape) accompanying the raw MEG recording.
  
## Project usage:
**src** contains all the code used for data preprocessing, analysis, and visualization.  
Use the codes under **src** for each analysis separately (General Decoding, Across Time Decoding, RSA, and Weight Projection). 

For all analyses:
  1. Open a project folder.
  2. Download the epoched data and store it in an "Evoked_fif" folder.
     Link for data download:  https://drive.google.com/drive/folders/17qm99CYq7jFmxApHIko3lCtprzKoUGvR?usp=drive_link
  3. Download the desired analysis codes and store them in the  "src" folder. 
  4. Change the project path in utils/config.py.
  5. Add the following folders to your project directory:
     * For general decoding (preprocessing):
         * For converting fif to npy files, create a directory for each time window: "BL_orig", "M100_orig"... "MLPP_orig" (each having a subject subdirectory)
         * For psuedo-trial calculation, create a directory for each size: "5PT" and "10PT" (each having a time window subdirectory)
         * Each time window subdirectory (e.g., "BL_Evoked_PPC, "M100_Evoked_PPC"... "MLPP_Evoked_PPC") should contain a subdirectory for each subject ("sub_003", "sub_004"... "sub_054")
     * For general decoding (analysis): "Results_general_decoding"
     * For across time decoding: "scores", "stats", "plots".
     * For RSA: "RSA_Results"
     * For weight projection: haufe_transform (for the activation weights)
  
Note: for weight projection, additionally download the raw data from - 
https://drive.google.com/drive/folders/16exK0kKPW0W8BUurdBni1D-hqHBp-To-?usp=drive_link.  
Store the raw data in the following manner: 


**Results** contains the results from general decoding, across time decoding, representational similarity analysis, and sensor weight projection
using the entire 42-subject dataset.
