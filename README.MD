# Planets Classification Project

This project is aimed at classifying satellite images of different types of land cover and land use. The project uses PyTorch Lightning for the model pipeline and ClearML for experiment tracking and logging. The models are versioned using DVC.

## Installation 
1. Clone the repository
```sh
git clone <your-repo-url>
cd <your-repo-name>
```
2. Install the required packages:
```sh
make install
```

## Download Dataset
To download and prepare the dataset, run:
```sh
make download_dataset
```
It will download the dataset and place it in the data folder located in the root directory of the project.

## Training the Model
To start training the model, use the following command:
```sh
make train
```

This will run the training process using the configuration specified in `config/config.yaml`.

## Experiment Tracking

We use ClearML for tracking experiments. You can find the metrics and configurations of the experiments in ClearML. Check out the experiment details [here](https://app.clear.ml/projects/f4150405bcd64ba99f413b9bf141dc40/experiments/ad104d9bbddd490f9da745f8025eb358/output/execution). Make sure you have your ClearML credentials set up.


## Model Checkpoints
Model checkpoints will be saved periodically during training in the `experiments` directory. These checkpoints can be used to resume training or for inference. The checkpoint files are named based on the epoch and validation F1 score.

## Resuming Training
If the training process is interrupted, you can resume training from the last checkpoint. 

## Model Versioning
We use DVC for model versioning. The latest and best model can be found in the DVC storage. To download the latest model weights from the remote storage, follow these steps:

1. **Configure the DVC remote storage**:
   Ensure your DVC remote storage is configured to point to the correct server. Run the following commands:

   ```bash
   dvc remote add --default alexander_vedernikov_remote ssh://91.206.15.25/home/a.vedernikov/dvc_files
   dvc remote modify alexander_vedernikov_remote user <ssh_username>
   dvc remote modify alexander_vedernikov_remote keyfile <path_to_private_key>
   ```
2. **Pull the latest weights**:
    Run the following command to pull the latest model weights from the remote storage:
    ```
    dvc pull experiments
    ```
3. **Verify the files**:
    ```
    ls experiments/experiment1
    ```
    You should see the model weight files such as `epoch_epoch=07-val_f1=0.618.ckpt`.

This will download the necessary files to your local setup, making it ready for inference or further training. For the inference, place the downloaded file into the `model_weights` folder.


## Running Inference
To run inference on new images, place the images in the `inference_images` folder. Then, use the following command from the root folder to perform inference:

```sh
make inference
```

This will run the inference process using the images in the `inference_images` folder and the latest model checkpoint. The results will be saved in the `predictions.csv` file in the root folder.


## Linting
We use wemake-python-styleguide for linting. The linter configurations are specified in the `src/setup.cfg` file. To run the linter, navigate to the `src` directory of the project and use the following command:

```sh
cd src
flake8 .
```
This will apply the linting rules defined in `setup.cfg` to the entire project.

## Repository Structure
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
├── Makefile
├── predictions.csv
├── README.md
└── requirements.txt
```


