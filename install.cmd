
echo :  update pip
python -m pip install --upgrade pip

echo :  install torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo :  updatetransformers
pip uninstall transformers -y
pip install --upgrade git+https://github.com/huggingface/transformers accelerate

echo :  install fastapi
pip install --upgrade -r requirements.txt