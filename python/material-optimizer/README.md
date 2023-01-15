## Material Optimizer (GUI)

### Student

Saip Can Hasbay, 01428723, University of Vienna, [saipcanhasbay@gmail.com](saipcanhasbay@gmail.com) or [a01428723@unet.univie.ac.at](a01428723@unet.univie.ac.at)

### Requirements:

- Python > 3.8
- CUDA:
  - GPU: NVidia driver >= 495.89 [(Taken from mitsuba3 requirements)](https://mitsuba.readthedocs.io/en/stable/#requirements)

### Dependencies:

- Please refer to _requirements.txt_ for required python packages

### Installation

1. Clone this repository
2. (optional) Create a virtual environment:
   1. Create:
      - ` python -m venv c:\path\to\myenv (e.g. python -m venv cloned_repository_root\myenv)`
   2. Activate your virtual environment:
      - `path_to_your_venv\Scripts\activate (e.g. myenv\Scripts\activate)`
3. Install dependencies
   - `pip install -r requirements.txt`

### Run

- If a virtual environment is in use, make sure to activate it (see [Installation 2.2](#installation))
- Run either through terminal:
  - `python material_optimizer_gui.py`
- Or IDE to your liking
  - e.g. vscode: Click to run icon on top right corner of _material_optimizer_gui.py_

### Optional data

- To retrieve additional [mitsuba-data](https://github.com/mitsuba-renderer/mitsuba-data) (e.g. test scenes provided from the mitsuba team), then run the following:
  - `git submodule update --init --recursive`

### Tips
- To change the current mitsuba version, modify the mitsuba line in `requirements.txt` and reinstall dependencies with:
  - `pip install -r requirements.txt`

### Known Issues

- After closing _material-optimizer_ sometimes Dr.Jit warns about leaked variables. For example:
  ```
  drjit-autodiff: variable leak detected (4 variables remain in use)!
   - variable a1154237 (1 references)
   - variable a1154245 (1 references)
   - variable a1154246 (1 references)
   - variable a1154244 (1 references)
  ```
  - Unfortunately the problem still persists, since manually clearning cache for Dr.Jit is not possible. Further analysis necessary.
- On mitsuba version 3.1.1 during optimization mitsuba - thus material-optimizer - might crash with the following error message:
  ```
  Critical Dr.Jit compiler failure: jit_optix_compile(): optixModuleGetCompilationState() indicates that the compilation did not complete succesfully. The module's compilation state is: 0x2363
  ```
  - The problem seems to be known and will be fixed in the upcoming mitsuba versions. [See Also](https://github.com/mitsuba-renderer/mitsuba3/issues/408)
