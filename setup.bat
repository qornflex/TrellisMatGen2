@echo off

REM ===================================================================================================================
REM  Created by: Quentin Lengele (16/10/2025)
REM ===================================================================================================================

REM ===================================================================================================================
REM PREREQUISITES
REM ===================================================================================================================

:: Install Python 3.10.x
::     https://www.python.org/ftp/python/3.10.9/python-3.10.9-amd64.exe

:: Install Git for Windows
::     https://gitforwindows.org

:: Install CUDA ToolKit 12.8
::     https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_571.96_windows.exe

:: Depends on 2 gated models (need access request):

::     https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
::     https://huggingface.co/briaai/RMBG-2.0

REM ===================================================================================================================

echo.
echo ===============================================
echo  Trellis.2 and MatGen installation for Windows
echo ===============================================
echo.

REM ===================================================================================================================
REM Ask for Python 3.10 folder
REM ===================================================================================================================

echo Enter the path to your Python 3.10 directory (e.g. C:\Python310\)
echo.
:get_python_path
set /p PYTHON_FOLDER=Python 3.10 Directory:

if not exist "%PYTHON_FOLDER%\python.exe" (
    echo.
	echo   Can't find any python executable here '%PYTHON_FOLDER%\python.exe'.
    echo.
	goto get_python_path
) else (
	set PYTHON_MAIN=%PYTHON_FOLDER%\python.exe
)

REM ===================================================================================================================
REM Check Python Version
REM ===================================================================================================================

for /f "tokens=2 delims= " %%v in ('%PYTHON_MAIN% --version') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do set PYSHORT=%%a.%%b

if "%PYSHORT%"=="3.10" (
    echo.
    echo   Python %PYVER% is installed.
	echo.
) else (
    echo.
    echo   =====================================================================================
    echo   The provided Python executable is %PYVER% (%PYTHON_MAIN%^)
    echo.
    echo   Please install Python 3.10:
    echo   https://www.python.org/ftp/python/3.10.9/python-3.10.9-amd64.exe
    echo   =====================================================================================
    echo.
    pause
    exit /b 1
)

REM ===================================================================================================================
REM Ask for CUDA ToolKit 12.8 folder
REM ===================================================================================================================

echo Enter the path to your CUDA Toolkit 12.8 directory
echo (e.g. C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\)
echo.
:get_cuda_path
set /p CUDA_FOLDER=Cuda TookKit Directory:

if not exist "%CUDA_FOLDER%\bin\nvcc.exe" (
    echo.
	echo   Can't find any CUDA executable here '%CUDA_FOLDER%\bin\nvcc.exe'.
    echo.
	goto get_cuda_path
) else (
	set CUDA_NVCC="%CUDA_FOLDER%\bin\nvcc.exe"
)

REM ===================================================================================================================
REM Check CUDA ToolKit 12.8 Installation
REM ===================================================================================================================

setlocal enabledelayedexpansion

set "CUDA_PATH=%CUDA_FOLDER%"
set "CUDA_HOME=%CUDA_FOLDER%"
set PATH=%CUDA_FOLDER%\bin;%CUDA_FOLDER%\libnvvp;%PATH%

nvcc --version >nul 2>&1
if %errorlevel%==0 (
    set "CUDA_VERSION="
    for /f "tokens=6" %%v in ('nvcc --version ^| findstr "release"') do set "CUDA_VERSION=%%v"

    rem Remove trailing comma
    set "CUDA_VERSION=!CUDA_VERSION:,=!"
    rem Remove leading V if present
    set "CUDA_VERSION=!CUDA_VERSION:V=!"
    rem Trim spaces
    for /f "tokens=* delims= " %%a in ("!CUDA_VERSION!") do set "CUDA_VERSION=%%a"

    rem Keep only major.minor
    set "CUDA_MAJOR=!CUDA_VERSION:~0,4!"

    if "!CUDA_MAJOR!"=="12.8" (
        echo.
        echo   CUDA Toolkit 12.8 is installed
        echo.
    ) else (
        echo.
        echo   =========================================================================================================
        echo   Your CUDA ToolKit version is !CUDA_VERSION!
        echo   You need to install CUDA Toolkit 12.8
        echo   Please install it from here:
        echo   https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_571.96_windows.exe
        echo   =========================================================================================================
        echo.
        pause
        exit /b 1
    )
) else (
    echo.
    echo   ===========================================================================================================
    echo   CUDA Toolkit 12.8 not installed or not found in your PATH environment variable
    echo   Please install it from here:
    echo   https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_571.96_windows.exe
    echo   ===========================================================================================================
    echo.
    pause
    exit /b 1
)

REM ===================================================================================================================
REM Check for Git Installation
REM ===================================================================================================================

REM Check if Git is available in PATH
echo Checking Git Installation...
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo   =====================================================================
    echo   Git is not installed or not found in your PATH environment variable.
    echo   Please install Git for Windows:
    echo   https://gitforwindows.org
    echo   =====================================================================
    echo.
    pause
    exit /b 1
)

git --version >nul 2>&1
if %errorlevel%==0 (
    REM Install and enable Git LFS
    echo   Git is installed
    git lfs install
	echo.
) else (
    echo.
    echo   =====================================================================
    echo   Git is not installed or not found in your PATH environment variable
    echo   Please install Git for Windows:
    echo   https://gitforwindows.org
    echo   =====================================================================
    echo.
    pause
    exit /b 1
)

REM -------------------------------------------------------------------------------------------------------------------
REM CLONE MODELS
REM -------------------------------------------------------------------------------------------------------------------

git clone https://huggingface.co/microsoft/TRELLIS.2-4B ./models/microsoft/TRELLIS.2-4B
git clone https://huggingface.co/microsoft/TRELLIS-image-large ./models/microsoft/TRELLIS-image-large

REM -------------------------------------------------------------------------------------------------------------------
REM VENV & PIP
REM -------------------------------------------------------------------------------------------------------------------

%PYTHON_MAIN% -m venv venv

call venv\Scripts\activate

call python -m pip install --upgrade pip
call python -m pip install wheel

REM -------------------------------------------------------------------------------------------------------------------
REM REQUIREMENTS
REM -------------------------------------------------------------------------------------------------------------------

REM PyTorch
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

REM Various dependencies
pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers gradio==6.0.1 tensorboard pandas lpips zstandard

REM hf_xet
pip install huggingface_hub[hf_xet]

REM utils3d
pip install ./tmp/wheels/utils3d-0.0.2-py3-none-any.whl

REM Blender
pip install bpy==4.0 --extra-index-url https://download.blender.org/pypi/

REM pillow
pip install pillow

REM kornia and timm
pip install kornia timm

REM pygltflib
pip install pygltflib

REM TRITON
pip install "triton-windows>=3.2.0,<3.6"

REM FLASH-ATTN
pip install https://huggingface.co/marcorez8/flash-attn-windows-blackwell/resolve/e1480e12fc744c1edf2f50831a5363d0faef45e4/flash_attn-2.7.4.post1-cp310-cp310-win_amd64-torch2.7.0-cu128/flash_attn-2.7.4.post1-cp310-cp310-win_amd64.whl

REM ----------------------------------------------------------------------------------------------------------------
REM NVIDIA FRAST
pip install ./tmp/wheels/nvdiffrast-0.4.0-cp310-cp310-win_amd64.whl

REM ----------------------------------------------------------------------------------------------------------------
REM NVDIFFREC
pip install ./tmp/wheels/nvdiffrec_render-0.0.0-cp310-cp310-win_amd64.whl

REM ----------------------------------------------------------------------------------------------------------------
REM CUMESH
pip install ./tmp/wheels/cumesh-0.0.1-cp310-cp310-win_amd64.whl

REM ----------------------------------------------------------------------------------------------------------------
REM FLEXGEMM
pip install ./tmp/wheels/flex_gemm-0.0.1-cp310-cp310-win_amd64.whl

REM ----------------------------------------------------------------------------------------------------------------
REM OVOXEL
pip install ./tmp/wheels/o_voxel-0.0.1-cp310-cp310-win_amd64.whl

echo.
echo ===============================
echo  Trellis MatGen 2.0 completed!
echo ===============================
echo.

pause