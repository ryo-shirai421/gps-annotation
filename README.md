# Context-Aware Semantic Annotation of Mobility Records

This is a reimplementation of the study published in the paper titled "Context-Aware Semantic Annotation of Mobility Records", which can be found [here](https://dl.acm.org/doi/10.1145/3477048).

## Setup Instructions

1. This project requires Python version 3.7.0.
2. Set up your local environment by running `poetry install`.
3. Generate the necessary data by creating a `data/` folder at the same level as the `src` directory and then running `notebooks/preprocess/pyspark2pandas.ipynb`.

## Running the Project
To execute the project, simply run the following command:

```
make run
```
This command will execute the main script of the project, training the model and producing the output.

## Project Structure

The project has the following directory structure:

```
.
├── src
│ ├── main.py
│ └── model
├── data
└── notebooks
　└── preprocess
    └── pyspark2pandas.ipynb
```


All models are stored in the `src/models` directory. To train a model, execute `src/main.py`. Model parameters are defined in the `cap_config.yaml` file.

## Data Information

The input data is divided into labeled and unlabeled data. After processing, the data format is as follows:

`labeled_df_formatted`:

| uid     | gps_lon   | gps_lat  | gps_timestamp | correct_category_index | feature_vec |
|---------|-----------|----------|---------------|------------------------|-------------|
| 0       | 139.0     | 35.0     | 14            | 25                     | [18, 9,... |
| 0       | 139.0     | 35.0     | 8             | 20                     | [69, 36,... |
| ...     | ...       | ...      | ...           | ...                    | ...         |

`unlabeled_df_formatted`:

| uid     | gps_lon   | gps_lat  | gps_timestamp |
|---------|-----------|----------|---------------|
| 14      | 139.0     | 35.0     | 0             |
| 14      | 139.0     | 35.0     | 12            |
| ...     | ...       | ...      | ...           |

