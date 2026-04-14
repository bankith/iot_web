#!/bin/bash  
  
# ── Conda ──  
source ~/miniconda3/etc/profile.d/conda.sh  
conda activate face_env  
  
# ── SNPE SDK ──  
export SNPE_ROOT=/home/ubuntu/2.27.0.240926  
export SNPE_TARGET_ARCH=aarch64-ubuntu-gcc9.4  
export LD_LIBRARY_PATH=$SNPE_ROOT/lib/$SNPE_TARGET_ARCH:$LD_LIBRARY_PATH  
export LD_LIBRARY_PATH="/home/ubuntu/snpe_dsp_libs:$LD_LIBRARY_PATH"
  
# ── Hexagon DSP ──  
export ADSP_LIBRARY_PATH="$SNPE_ROOT/lib/hexagon-v68/unsigned;."  
export ADSP_LIBRARY_PATH="/home/ubuntu/2.27.0.240926/lib/hexagon-v68/unsigned;/usr/share/qcom/qcm6490/Thund>
  
# ── Python paths ──  
# libsnpehelper.so (SNPE Python extension)  
SNPE_HELPER_DIR=$(find $SNPE_ROOT -name "libsnpehelper.so" -exec dirname {} \; | head -1)  
if [ -n "$SNPE_HELPER_DIR" ]; then  
    export PYTHONPATH="$SNPE_HELPER_DIR:$PYTHONPATH"  
    echo "✓ libsnpehelper found: $SNPE_HELPER_DIR"  
else  
    echo "⚠ libsnpehelper.so not found in SNPE_ROOT"  
fi  
  
# Repo root (so web2.py can find snpehelper_manager.py)  
export PYTHONPATH="/home/ubuntu/finalProject2:$PYTHONPATH"  
  
echo "✓ SNPE_ROOT=$SNPE_ROOT"  
echo "✓ LD_LIBRARY_PATH includes $SNPE_ROOT/lib/$SNPE_TARGET_ARCH"