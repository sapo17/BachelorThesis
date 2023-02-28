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
      - Windows: `path_to_your_venv\Scripts\activate (e.g. myenv\Scripts\activate)`
      - Linux: `source path_to_your_venv\bin\activate (e.g. source myenv\bin\activate)`
3. Install dependencies
   
   0. (optional): Make sure pip is updated
      - `pip install --upgrade pip`
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
- Change mitsuba version by modifiying the mitsuba row in the `requirements.txt` file and following the [installation](#installation) steps.
- On Visual Studio Code we suggests using the Python extension by Microsoft.
- You can check the availability of CUDA with the following command:
  - `nvcc --version`
  - Which should respond with something like this:
    ```
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2022 NVIDIA Corporation
    Built on Wed_Jun__8_16:49:14_PDT_2022
    Cuda compilation tools, release 11.7, V11.7.99
    Build cuda_11.7.r11.7/compiler.31442593_0
    ```
- If CUDA is not available in your system, or you receive an error similar to this:
  
  ```
    AttributeError: jit_init_thread_state(): the CUDA backend is inactive because it has not been initialized via jit_init(), or because the CUDA driver library ("libcuda.so") could not be found! Set the DRJIT_LIBCUDA_PATH environment variable to specify its path.
  ```
  - **Linux**: You might fix the problem with adding the following lines (_if CUDA is indeed installed in your system_) to your _.bashrc_ file:
    ```
    # cuda path
    export PATH="/usr/local/cuda-12.0/bin:$PATH" # set according to cuda version
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
    ```
  - Otherwise you might navigate to the `./src/material_optimizer_model.py` file and replace the mitsuba variant as demonstrated below. Although at the moment we don't support LLVM (CPU) variant of mitsuba, it still might help to run the application.
    ```
      # set mitsuba variant: NVIDIA CUDA
      # mi.set_variant(CUDA_AD_RGB) # replace this line with
      mi.set_variant('llvm_ad_rgb') # with this
    ```
- Dr.Jit Cache location: `C:\Users\user\AppData\Local\Temp\drjit\`
- Automatic UML Model generation. See [also](https://www.bhavaniravi.com/python/generate-uml-diagrams-from-python-code).
  - ```pyreverse -o svg --colorized -p MaterialOptimizer .``` (make sure "." refers to this repository location)
- Beware that under linux you may need to change some path/s in provided Mitsuba XML files. For example, the file under `scenes/material-preview/translucent-principled-bsdf/scene-init-bunny.xml`, beginning for line 80:
  ```
    <shape type="ply">
        <string name="filename" value="scenes\material-preview\meshes\bunny.ply"/> # change this
        <string name="filename" value="scenes/material-preview/meshes/bunny.ply"/> # with this
		<transform name="to_world">
			<scale value="13"/>
			<rotate z="1" angle="130"/>
			<rotate y="1" angle="70"/>
			<rotate x="1" angle="-50"/>
			<rotate z="1" angle="-50"/>
			<rotate y="1" angle="-2"/>
			<rotate x="1" angle="12"/>
			<translate x="0.3" y="0.0" z="-0.5"/>
		</transform>
        <ref id="object_bsdf"/>
    </shape>
  ```
- Under videos directory you may find simple optimization examples.


### Known Issues

- After closing _material-optimizer_ sometimes Dr.Jit warns about leaked variables. For example:
  ```
  drjit-autodiff: variable leak detected (4 variables remain in use)!
   - variable a1154237 (1 references)
   - variable a1154245 (1 references)
   - variable a1154246 (1 references)
   - variable a1154244 (1 references)
  ```
  - Unfortunately the problem still persists, since programatically clearning cache for Dr.Jit is not possible. Further analysis necessary.
- On mitsuba version 3.1.1 during optimization mitsuba - thus material-optimizer - might crash with the following error message:
  ```
  Critical Dr.Jit compiler failure: jit_optix_compile(): optixModuleGetCompilationState() indicates that the compilation did not complete succesfully. The module's compilation state is: 0x2363
  ```
  - The problem seems to be known and will be fixed in the upcoming versions. [See Also](https://github.com/mitsuba-renderer/mitsuba3/issues/408)

