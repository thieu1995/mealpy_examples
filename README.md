# Mealpy Examples

```code 
Dataset

https://neptune.ai/blog/select-model-for-time-series-prediction-task

```

## Environment

1. Using conda/miniconda to create new environment

```code

   conda create -n new ml python==3.8.5
   conda activate ml
   
   conda install -c conda-forge numpy
   conda install -c conda-forge pandas
   conda install -c conda-forge scikit-learn
   conda install -c conda-forge matplotlib
   conda install -c conda-forge tensorflow==2.9.1
   conda install -c conda-forge keras==2.9.0
   
   pip install mealpy==2.4.1
   pip install permetrics==1.2.2
```

2. Using pip to create new environment

```code 
============ Install environment on Windows ===============
open terminal: (pip on windows is already installed inside python)
   python -m venv pve_paper
   pve_paper\Scripts\activate.bat
   pip install -r requirements.txt
   
   
============ Install environemtn on Ubuntu ==================

sudo apt-get install pip3
sudo apt-get install python3.8-venv

python3 -m venv pve_paper 
source pve_paper/bin/activate
pip install -r requirements.txt

deactivate (stop using environment)

```


## Pip tutorials

```code 

pip list (pip auto-downloaded in python)

pip list --local (only packages installed from current environment)


1. Create a blank environment 

python -m venv env_name (create a blank environment named "env_name")

env_name\Scripts\activate.bat (activate env)

pip freeze > requirements.txt 


2. Create an inheritance environment 

python -m venv india --system-site-packages (create india environment inherits packages from based-system)

india\Scripts\activate.bat  (activate india env)

pip install numpy (install package to current env)

pip freeze --local > requirements.txt (Create requirements file that installed additional packages)

rmdir india /s 	(remove all environment)


3. Remove pip from python 

python -m pip uninstall pip

4. Export environments 

pip list --format=freeze > requirements.txt 

pip freeze --local > requirements.txt

```

## Helper

https://pythontic.com/modules/pickle/dumps
https://medium.com/fintechexplained/how-to-save-trained-machine-learning-models-649c3ad1c018
https://github.com/keras-team/keras/issues/14180
https://ai-pool.com/d/how-to-get-the-weights-of-keras-model-

```python 
https://stackoverflow.com/questions/1894269/how-to-convert-string-representation-of-list-to-a-list

import json 
x = "[0.7587068025868327, 1000.0, 125.3177189672638, 150, 1.0, 4, 0.1, 10.0]"
solution = json.loads(x)
print(solution)

```


