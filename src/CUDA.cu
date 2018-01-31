#include <cuda.h>
#include "../include/CUDA_wrappers.hpp"
#include <host_defines.h>
#include <device_launch_parameters.h>

// функция вычисления точки
__global__ void compute_point(double* x_down, double* x_up, double* y_down, double* y_up, unsigned int* iteration_count, unsigned int* width, unsigned int* height, unsigned int* matrix) {
    double re = *x_down + (*x_up - *x_down) * ((blockIdx.x % *width) + 0.5) / (double) *width;
    double im = *y_up - (*y_up - *y_down) * ((blockIdx.x / *width) + 0.5) / (double) *height;
    double re_curr = 0;
    double im_curr = 0;
    // проверка на принадлежность главной картиоиде

    // проверка точки
    for(unsigned int iteration = 1; iteration <= *iteration_count; ++iteration) {
        re_curr = re_curr * re_curr - im_curr * im_curr + re;
        im_curr = 2 * re_curr * im_curr + im;
        if(re_curr * re_curr + im_curr * im_curr >= 4.0) {
            matrix[blockIdx.x] = iteration;
            break;
        }
    }
    matrix[blockIdx.x] = 0; // если точка все еще не вышла за границу
}

// функция, которая будет запускать вычисления
void compute_matrix(unsigned int* matrix, double x_down, double x_up, double y_down, double y_up, unsigned int width, unsigned int height, unsigned int iteration_count) {
    // просчитываем изображение на CUDA
    // выделяем память на видеокарте
    double* dev_x_down;
    double* dev_x_up;
    double* dev_y_down;
    double* dev_y_up;
    unsigned int* dev_iteration_count;
    unsigned int* dev_width;
    unsigned int* dev_height;
    unsigned int* dev_matrix;
    cudaMalloc((void**)&dev_x_down, sizeof(double));
    cudaMalloc((void**)&dev_x_up, sizeof(double));
    cudaMalloc((void**)&dev_y_down, sizeof(double));
    cudaMalloc((void**)&dev_y_up, sizeof(double));
    cudaMalloc((void**)&dev_iteration_count, sizeof(unsigned int));
    cudaMalloc((void**)&dev_width, sizeof(unsigned int));
    cudaMalloc((void**)&dev_height, sizeof(unsigned int));
    cudaMalloc((void**)&dev_matrix, width * height * sizeof(unsigned int));
    // переносим данные в память на видеокарте
    cudaMemcpy(dev_x_down, &x_down, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x_up, &x_up, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y_down, &y_down, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y_up, &y_up, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_iteration_count, &iteration_count, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_width, &width, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_height, &height, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrix, &matrix, width * height * sizeof(unsigned int), cudaMemcpyHostToDevice);
    // запускаем ядро
    compute_point<<< (width * height), 1 >>>(dev_x_down, dev_x_up, dev_y_down, dev_y_up, dev_iteration_count, dev_width, dev_height, dev_matrix);
    // event
    cudaEvent_t syncEvent;
    cudaEventCreate(&syncEvent);    //Создаем event
    cudaEventRecord(syncEvent, 0);  //Записываем event
    cudaEventSynchronize(syncEvent);  //Синхронизируем event
    // забираем данные из памяти видеокарты
    cudaMemcpy(matrix, dev_matrix, width * height, cudaMemcpyDeviceToHost);
    // освобождаем память на видеокарте
    cudaFree(dev_x_down);
    cudaFree(dev_x_up);
    cudaFree(dev_y_down);
    cudaFree(dev_y_up);
    cudaFree(dev_iteration_count);
    cudaFree(dev_width);
    cudaFree(dev_height);
    cudaFree(dev_matrix);
}
