# Training set

The training set should be obtained from the `midi_preprocess/data_biLstm` folder and it should be put in the `notes` folder.

# Training

Please specify the training set and the number of epochs as two arguments.
for example: `python train.py beethoven_notes 20`

# Generation

Please specify the training set, the optimal weight, and the number of notes as three arguments.
for example: `python generate.py beethoven_notes file_name_of_the_weight 100`