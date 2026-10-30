# Hall 2018 calibration input

Hall et al. (2018), Glucotypes reveal new patterns of glucose dysregulation,
PLOS Biology 16(7): e2005143. https://doi.org/10.1371/journal.pbio.2005143

Official article and supporting data:
https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.2005143

cohort.py reads data/public/Hall2018_glucotypes/S5_Data_clinical.sqlite,
table clinical, columns userID, FBG, mean_glucose and sd_glucose. It selects a
development subset before fitting a fasting-to-mean glucose bridge. The source
is a small short-record cohort. The code explicitly leaves bridge time alignment
unverified; a cross-sectional bridge is not a validated longitudinal mechanism.
Participant-level SQLite/CSV files are not republished.
