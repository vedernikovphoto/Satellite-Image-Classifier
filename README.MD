# Planets Classification Project
This project focuses on classifying satellite images based on different types of land cover and land use. The implementation uses PyTorch Lightning for model training pipelines, ClearML for experiment tracking, and DVC for model versioning and data management.


## Installation
1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. **Install the required packages:**
    ```sh
    make install
    ```


## Download Dataset
To download and prepare the dataset, run the following command:
```sh
make download_dataset
```
This command will download the dataset and place it in the `data` directory located in the root of the project.


## Training the Model
To start training the model, use the following command:
```sh
make train
```
This command will initiate the training process using the configuration specified in `config/config.yaml`.


## Experiment Tracking
We use ClearML for tracking experiments. You can view the metrics and configurations of the experiments on ClearML. Access the experiment details [here](https://app.clear.ml/projects/f4150405bcd64ba99f413b9bf141dc40/experiments/ad104d9bbddd490f9da745f8025eb358/output/execution). 

Ensure that your ClearML credentials are properly configured before running the experiments.


## Model Checkpoints
Model checkpoints are saved periodically during training in the `experiments` directory. These checkpoints can be used to resume training or for inference. The checkpoint files are named based on the epoch and validation F1 score.


## Resuming Training
If the training process is interrupted, you can resume training from the last saved checkpoint. Ensure that the checkpoint file is located in the `experiments` directory, then restart the training using the appropriate command.


## Model Versioning

We use DVC for model versioning. The model weights are stored in a private remote storage configured through DVC. If you have access to this storage, follow the steps below to retrieve the weights:

1. **Configure the DVC remote storage:**
   Ensure your DVC remote storage is correctly configured. Run the following commands:

   ```bash
   dvc remote add --default <remote_name> <remote_url>
   dvc remote modify <remote_name> user <username>
   dvc remote modify <remote_name> keyfile <path_to_private_key>
    ```

2. **Pull the latest weights:**
    Run the following command to pull the latest model weights from the remote storage:
    ```bash
    dvc pull experiments
    ```

3. **Verify the files:** 
    After pulling the files, verify them using:
    ```bash
    ls experiments/experiment1
    ```
    You should see the model weight files such as `model.pt`.

This process ensures that you have the necessary files in your local setup for inference or further training. For inference, place the downloaded file into the `model_weights` directory.

If you do not have your own DVC remote storage, you can request the model weights by sending an email to [alexander.vedernikov.edu@gmail.com](mailto:alexander.vedernikov.edu@gmail.com). Once you receive the weights, place them in the `model_weights` directory.


## Running Inference

To run inference on new images, place the images in the `inference_images` directory. Then, execute the following command from the root of the project:

```sh
make inference
```

This will perform inference using the images in the `inference_images` directory and the latest model checkpoint. The results will be saved in the `predictions.csv` file in the root directory.


## Linting

We use `wemake-python-styleguide` for linting. The linter configurations are specified in the `src/setup.cfg` file. To run the linter, navigate to the `src` directory of the project and use the following command:

```sh
cd src
flake8 .
```

This command will apply the linting rules defined in `setup.cfg` to the entire project.


## Repository Structure

The following is an overview of the project's directory structure:

```
├── config
│   └── config.yaml
├── inference_images
│   ├── file_99.jpg
│   ├── file_212.jpg
│   └── ...
├── model_weights
├── src
│   ├── augmentations.py
│   ├── config.py
│   ├── constants.py
│   ├── datamodule.py
│   ├── dataset.py
│   ├── dataset_splitter.py
│   ├── lightning_module.py
│   ├── losses.py
│   ├── metrics.py
│   ├── setup.cfg
│   ├── train.py
│   └── train_utils.py
├── .gitignore
├── Makefile
├── predictions.csv
├── README.md
└── requirements.txt
```
