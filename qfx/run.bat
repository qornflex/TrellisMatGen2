@echo off

REM ===================================================================================================================
REM  Created by: Quentin Lengele (16/10/2025)
REM ===================================================================================================================

set PY_LIB_PATH=%1
set PY_FILE=%2
set INPUT_FILELIST=%3

set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0
set OPENCV_IO_ENABLE_OPENEXR=1
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.3,max_split_size_mb:128

set PYTHONPATH=%PY_LIB_PATH%;%PY_LIB_PATH%\matgen;

cd /D "%PY_LIB_PATH%"
call venv\Scripts\activate

call python qfx/%PY_FILE% %INPUT_FILELIST%
