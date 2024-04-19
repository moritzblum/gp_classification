
# Can GNNs approximate graph pattern matching based graph classification?



![alt text](https://anonymous.4open.science/api/repo/gp_classification-81B8/file/figures/enriched-DLM-example.png)


### Set-up a Conda Environment and Install Dependencies
```bash
conda create --name gp python=3.10
conda activate gp
pip install -r requirements.txt
```
If CUDA GPUs are availible, the corresponding Torch versions are automatically installed. 

### Set-up Project Structure
```bash
git clone git@github.com:moritzblum/gp_classification.git
cd gp_classification
```

### Dataset 
Run the follwing commands to downlaod the dataset from [Zenodo](https://doi.org/10.5281/zenodo.10988584) and extract it to the `data` folder. 
```bash
cd data
wget https://zenodo.org/records/10988584/files/zenodo-synthetic-2-5.zip
unzip zenodo-synthetic-2-5.zip -d raw/
cd ..
```

This script creates a folder `data` with a sub-folder `raw` in the root directory. It places all files from [Zenodo](https://doi.org/10.5281/zenodo.10988584) in the `data/raw` directory. 

When running the training script, the data is automatically preprocessed and saved in the `data` folder. Therefore, when you make changes 
to the raw data, you need to delete all the data in the `data` folder to avoid loading old data and to 
allow re-pre-processing.

### Run Experiments
The bash script `run_gp.sh` contains all the commands to run the experiments. 


### Reported Results 
The training/validation/test result logs from our experiments can be found in the folder `results`. The file names begin with *"paper="*, e.g., `paper=01_model=GCN_freeze=False_dim=32_lr=0.001_ming=75.json`.




