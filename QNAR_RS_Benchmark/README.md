# Recommendation System Benchmarking

In order to run the benchmarking experiments, you need to first convert the datasets into atomic files
so they can be used by RecBole. We currently use three datasets for benchmarking:

- YooChoose
- Diginetica
- Quantinar Dataset

The first two help us to test the performance of our recommendation system on a real-world dataset.
The third one is a specific dataset that the model was especially developed for and it is used to point out the 
performance of our model against other models.

## Datasets

In order to convert the datasets into atomic files, you first need to download them from theit respective websites as specific
at [Dataset List](https://recbole.io/dataset_list.html), and place them in ../data folder. After that, you need to download the 
RecDatasets converter and install the requirements.

```bash
git clone https://github.com/RUCAIBox/RecDatasets.git
cd RecDatasets/conversion_tools
pip install -r requirements.txt
```

Then, you can run the following commands to convert the datasets. Specific for each dataset.

### YooChoose

YooChoose is a dataset that contains information about the courses that a user has taken and the courses that he/she
wants to take.

### Diginetica

Converting Diginetica dataset to atomic files.
```bash
python RecDatasets/conversion_tools/run.py --dataset diginetica --input_path ../data/diginetica/dataset-train-diginetica --output_path data/diginetica --duplicate_removal --convert_inter
python RecDatasets/conversion_tools/run.py --dataset diginetica --input_path ../data/diginetica/dataset-train-diginetica --output_path data/diginetica --convert_item
```