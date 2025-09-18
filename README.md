# Segmentations utilities

## Get the data
- PFCS data: `git clone git@github.com:PeARL-robotics/PFCS.git`
- Cleaner table task data can be downloaded from: `https://studentuml-my.sharepoint.com/:f:/r/personal/andrea_pierre_student_uml_edu/Documents/table-task?csf=1&web=1&e=nAn7l5` (need to be granted access, don't forget the JSON file).

## Data exploration
To have a look at the recorded demonstration data, run the following script: `panel serve PFCS_synchro_video_eef_position.py`, and open the URL (e.g. http://localhost:5006/ground_truth_segm_synchro_video_eef_position) in your browser.

## Ground truth segmentation
To get the ground truth segmentation data, move the slider and save the Unix epoch number between each segment in the JSON file.
