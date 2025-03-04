## Setup the conda environment using the yml file

```sh
conda env create -f fileName.yml

or

conda env create --file=URL
```

## Creating the similar environment with python virtual env

```sh
python -m venv llms-env

# Windows
llms-env\Scripts\activate

# MAC
source llms-env/bin/activate

# Install everything from the requirements.txt file
pip install -r requirements.txt
```
