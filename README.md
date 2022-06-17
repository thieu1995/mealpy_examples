# Mealpy Examples

## Environment

1. If using conda 
```code
+ Should create a new environment:
   conda create -n new ml python==3.7.5
   conda activate ml
   
   conda install -c conda-forge numpy
   conda install -c conda-forge pandas
   conda install -c conda-forge scikit-learn
   conda install -c conda-forge matplotlib
   conda install -c conda-forge tensorflow==2.1.0
   conda install -c conda-forge keras==2.3.1
   
   pip uninstall mealpy
   pip uninstall permetrics
   pip install mealpy==2.4.0
   pip install permetrics==1.2.2
```

2. If using pip 

```code 
+ Should create a new environment:
   python -m venv ml
   ml\Scripts\activate.bat
   pip install -r requirements.txt
```


## Export environment

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


