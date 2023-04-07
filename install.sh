conda create -n colink-protocol-unifed-tff python=3.9.16 -y
conda activate colink-protocol-unifed-tff
pip install --upgrade pip
pip install tensorflow-federated==0.42.0
pip install tensorflow-probability==0.15.0
pip install flbenchmark
pip install nest_asyncio
pip install pillow
pip install tensorflow-addons==0.19.0
pip install scikit-learn
pip install pytest
pip install -e .
