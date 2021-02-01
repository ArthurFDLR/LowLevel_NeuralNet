# C++ build
build-lib:
	g++ -Wall LowLevel_NeuralNet/evaluation_cpp/src/*.cpp -fPIC -O3 -g -shared -o LowLevel_NeuralNet/evaluation_cpp/libllnn.so

build-test: build-lib
	g++ -Wall LowLevel_NeuralNet/evaluation_cpp/llnn_test.cpp -g -lm -L./LowLevel_NeuralNet/evaluation_cpp -I./LowLevel_NeuralNet/evaluation_cpp -Wl,-rpath=. -lllnn -o LowLevel_NeuralNet/evaluation_cpp/llnn_test

# Python formatting
fmt-black:
	python -m black LowLevel_NeuralNet/ tests/

lint:
	python -m black --check LowLevel_NeuralNet/ tests/

# Tests
dev-env:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

speed-test:
	python ./tests/speed.py