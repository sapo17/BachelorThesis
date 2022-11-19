## Material Optimizer (GUI)
### Student
Saip Can Hasbay, 01428723, University of Vienna, [saipcanhasbay@gmail.com](saipcanhasbay@gmail.com) or [a01428723@unet.univie.ac.at](a01428723@unet.univie.ac.at)

### Requirements:
- Python > 3.8
- CUDA:
    - GPU: NVidia driver >= 495.89 [(Taken from mitsuba3 requirements)](https://mitsuba.readthedocs.io/en/stable/#requirements)

### Dependencies:
- Please refer to *requirements.txt* for required python packages

### Installation
1. Clone this repository
2. (optional) Create a virtual environment:
    1. Create:
        - ``` python -m venv c:\path\to\myenv (e.g. python -m venv cloned_repository_root\myenv)```
    2. Activate your virtual environment:
        - ``` path_to_your_venv\Scripts\activate (e.g. myenv\Scripts\activate) ```
3. Install dependencies
    - ``` pip install -r requirements.txt ```

### Run
- If a virtual environment is in use, make sure to activate it (see [Installation 2.2](#installation))
- Run either through terminal:
    - ``` python material_optimizer.py ```
- Or IDE to your liking
    - e.g. vscode: Click to run icon on top right corner of *material_opt_gui.py*

### Optional data
- To retrieve additional [mitsuba-data](https://github.com/mitsuba-renderer/mitsuba-data) (e.g. test scenes provided from the mitsuba team), then run the following:
  - ``` git submodule update --init --recursive ```