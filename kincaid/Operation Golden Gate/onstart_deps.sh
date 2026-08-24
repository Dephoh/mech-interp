#!/bin/bash
exec > /workspace/setup.log 2>&1
set -x
pip install -q --upgrade pip
pip install -q "numpy<2" transformers accelerate huggingface_hub
python -c "import torch,transformers,numpy;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available());print(\"np\",numpy.__version__,\"tf\",transformers.__version__)"
echo SETUP_DONE
