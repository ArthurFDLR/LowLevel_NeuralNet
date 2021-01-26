# C++ build
build_lib:
	g++ -Wall LowLevel_NeuralNet/evaluation_cpp/src/*.cpp -fPIC -O -g -shared -o LowLevel_NeuralNet/evaluation_cpp/libllnn.so

build_test: build_lib
	g++ -Wall LowLevel_NeuralNet/evaluation_cpp/llnn_test.cpp -g -lm -L./LowLevel_NeuralNet/evaluation_cpp -I./LowLevel_NeuralNet/evaluation_cpp -Wl,-rpath=. -lllnn -o LowLevel_NeuralNet/evaluation_cpp/llnn_test

# Python formatting
fmt-black:
	python -m black LowLevel_NeuralNet/ tests/

lint:
	python -m black --check LowLevel_NeuralNet/ tests/

# Tests
editable_package:
	python -m pip install -e .

test: build_lib editable_package
	python -m pytest