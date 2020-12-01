# explainable_concept_drift_pm
This is the implementaiton of the Framework for Explainable Concept Drift Detection for Process Mining Paper, built ontop of PM4Py https://pm4py.fit.fraunhofer.de/.
## Installation
Under Python 3.7, use pip to install the requirements
```bash
pip install -r requirements.txt
```
If you use anaconda just run the commands specified in setup.txt, this will setup a new environment under pyhton 3.7 and instlal all required packages.

## Usage
To run the two examples specified in the paper, run either
```bash
python synthetic.py
```
or 
```bash
python bpi_2017.py
```
The output will be written to the corresponding pdfs.

In order to run the bpi_2017.py script, please unzip the log in pm4py/statistics/time_series/experiments/data into BPI2017.xes

