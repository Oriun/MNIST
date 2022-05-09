# MNIST
Handwritten digits recognition solved with various algorithms for a understanding of artificial intelligence

### Start Sci-kit learn (python) version

```shell
cd sklearn
pip install -r requirements.txt
py main.py
```
### Start Custom version

```shell
cd custom
sh _prepare.sh # unzip dataset and move it to a `data`directory
g++ main.cpp --std=c++17 -o knn # compile the c++ code
./knn # execution
```