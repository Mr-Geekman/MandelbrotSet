#include <host_defines.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cmath>
#include "../include/CUDA_wrappers.hpp"

#define PRECISION double

// функция вычисления точки
__global__ void compute_point(PRECISION* x_down, PRECISION* x_up, PRECISION* y_down, PRECISION* y_up, unsigned int* iteration_count, unsigned int* width, unsigned int* height, unsigned int* matrix) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    PRECISION re = *x_down + (*x_up - *x_down) * (col + 0.5) / (PRECISION) (*width);
    PRECISION im = *y_up - (*y_up - *y_down) * (row + 0.5) / (PRECISION) (*height);
    PRECISION re_curr = 0.0;
    PRECISION im_curr = 0.0;
    // проверка на принадлежность главной картиоиде

    // проверка точки
    if(row >= *height || col >= *width) {
        return;
    }
    PRECISION re_temp;
    for(unsigned int iteration = 1; iteration <= *iteration_count; ++iteration) {
        re_temp = re_curr * re_curr - im_curr * im_curr + re;
        im_curr = 2.0 * re_curr * im_curr + im;
        re_curr = re_temp;
        if(re_curr * re_curr + im_curr * im_curr >= 4.0) {
            matrix[row * (*width) + col] = iteration;
            return;
        }
    }
    matrix[row * (*width) + col] = 0; // если точка все еще не вышла за границу
}

// функция, которая будет запускать вычисления
void compute_matrix(unsigned int* matrix, PRECISION x_down, PRECISION x_up, PRECISION y_down, PRECISION y_up, unsigned int width, unsigned int height, unsigned int iteration_count) {
    // просчитываем изображение на CUDA
    // выделяем память на видеокарте
    PRECISION* dev_x_down;
    PRECISION* dev_x_up;
    PRECISION* dev_y_down;
    PRECISION* dev_y_up;
    unsigned int* dev_width;
    unsigned int* dev_height;
    unsigned int* dev_matrix;
    unsigned int* dev_iteration_count;
    cudaMalloc((void**)&dev_x_down, sizeof(PRECISION));
    cudaMalloc((void**)&dev_x_up, sizeof(PRECISION));
    cudaMalloc((void**)&dev_y_down, sizeof(PRECISION));
    cudaMalloc((void**)&dev_y_up, sizeof(PRECISION));
    cudaMalloc((void**)&dev_width, sizeof(unsigned int));
    cudaMalloc((void**)&dev_height, sizeof(unsigned int));
    cudaMalloc((void**)&dev_iteration_count, sizeof(unsigned int));
    cudaMalloc((void**)&dev_matrix, width * height * sizeof(unsigned int));
    // переносим данные в память на видеокарте
    cudaMemcpy(dev_x_down, &x_down, sizeof(PRECISION), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x_up, &x_up, sizeof(PRECISION), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y_down, &y_down, sizeof(PRECISION), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y_up, &y_up, sizeof(PRECISION), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_width, &width, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_height, &height, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_iteration_count, &iteration_count, sizeof(unsigned int), cudaMemcpyHostToDevice);
    // запускаем ядро
    dim3 block_size(16, 16);
    dim3 grid_size(ceil((double) width / (double) block_size.x), ceil((double) height / (double) block_size.y));
    compute_point<<<grid_size, block_size>>>(dev_x_down, dev_x_up, dev_y_down, dev_y_up, dev_iteration_count, dev_width, dev_height, dev_matrix);
    // забираем данные из памяти видеокарты
    cudaMemcpy(matrix, dev_matrix, width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    // освобождаем память на видеокарте
    cudaFree(dev_x_down);
    cudaFree(dev_x_up);
    cudaFree(dev_y_down);
    cudaFree(dev_y_up);
    cudaFree(dev_width);
    cudaFree(dev_height);
    cudaFree(dev_iteration_count);
    cudaFree(dev_matrix);
}
