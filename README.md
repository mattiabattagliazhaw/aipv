# aipv
Code related to paper "Deep Ensemble Inverse Model for Image-Based Estimation of Solar Cell Parameters"

The results described in the paper can be reproduced as follows:

1. Run data_generation/generate_laoss_images.py. The script will generate Laoss simulations in the folder example_data
and transform them to the input image format described in the paper. Change the number of simulated images (n_images = 100)
on line 56 to a value suitable for your computational resources. (Requires a Laoss license.)
2. Run data_generation/export_training_data.oy. The script will collect all simulated images, process them and store
them in x_train.npy and x_test.npy. The corresponding model target values are stored in y_train.txt and y_test.txt.
3. Start the model training by running model_training/aipv_model_training.py

The full dataset used for the results in the paper are available at: https://drive.switch.ch/index.php/s/hI4sblyhz9FGBlf
