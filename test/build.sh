g++ -std=c++2a \
    -I/home0/y2024/u2424004/.local/mycudnn/TensorRT-6.0.1.5/include \
    -L/home0/y2024/u2424004/.local/mycudnn/TensorRT-6.0.1.5/lib \
    suiron.cpp -o suiron \
    -lnvinfer -lnvonnxparser -lcudart \
    -W
