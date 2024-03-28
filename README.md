
# Can GNNs approximate graph pattern matching based graph classification?

### Set-up a Conda Environment and Install Dependencies
```bash
conda create --name gp python=3.10
conda activate gp
pip install -r requirements.txt
```
If CUDA GPUs are availible, the corresponding Torch versions are automatically installed. 

### Dataset 
Create a folder `data` with a sub-folder `raw` in the root directory:
```
mkdir data
mkdir data/raw
```
Place all files in the `data/raw` directory. When running the training script, the data is automatically 
preprocessed and saved in the `data` folder. Therefore, when you make changes 
to the raw data, you need to delete all the data in the `data` folder to avoid loading old data and to 
allow re-pre-processing.



### Run Experiments
The bash script `run_gp.sh` contains all the commands to run the experiments. 
