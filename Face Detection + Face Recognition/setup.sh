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
export ADSP_LIBRARY_PATH="/home/ubuntu/2.27.0.240926/lib/hexagon-v68/unsigned;/usr/share/qcom/qcm6490/Thundercomm/RB3gen2/dsp/cdsp;/usr/lib/rfsa/adsp/cdsp;/usr/lib/dsp/cdsp;."  
  
# ── Python paths ──  
export PYTHONPATH="/home/ubuntu/finalProject2:$PYTHONPATH"  
  
echo "✓ SNPE_ROOT=$SNPE_ROOT"  
echo "✓ LD_LIBRARY_PATH includes $SNPE_ROOT/lib/$SNPE_TARGET_ARCH"  
echo "✓ ADSP_LIBRARY_PATH=$ADSP_LIBRARY_PATH"  