#include <iostream>
#include <cstdio>
#include <array>
#include <complex>
#include <cmath>
#include <ctime>
#include <gd.h>
#include <cuda_runtime_api.h>
#include "include/CUDA_wrappers.hpp"

void draw_zoom_gif(double x, double y, double zoom_coefficient, int num_of_zooms, unsigned int width, unsigned int height, unsigned int iteration_count);
// рисует gif, который будет приближаться к определенной точке
void draw_single_bmp(std::array<double, 4> size, unsigned int width, unsigned int height, unsigned int iteration_count);
// рисует одну bmp картинку
void draw_single_gif(std::array<double, 4> size, unsigned int width, unsigned int height, unsigned int iteration_count);
// рисует одну gif картинку
void draw_mandelbrot_set(gdImagePtr& image, std::array<double, 4> size, unsigned int width, unsigned int height, unsigned int iteration_count);
// рисует множество Мандельброта
unsigned int check_point(double x, double y, unsigned int iteration_count);
// проверяет точку на принадлежность множеству Мандельброта, возвращает 0, если принадлежит и число итераций, если не принадлежит
void paint_point(gdImagePtr& image, int x, int y, unsigned int iteration);
// красит точку в соответсвии с ее параметрами

void draw_zoom_gif(double x, double y, double zoom_coefficient, int num_of_zooms, unsigned int width, unsigned int height, unsigned int iteration_count) {
    gdImagePtr image;
    FILE* fout;
    fout = fopen("/home/mrgeekman/Документы/Программирование/С++/Программы/Mandelbrot set/Media/Mandelbrot_set_zoom.gif", "wb");
    std::array<double, 4> size;
    size[0] = x - 2.0; // нижняя граница x
    size[1] = x + 2.0; // верхняя граница x
    size[2] = y - 2.0; // нижняя граница y
    size[3] = y + 2.0; // верхняя граница y
    draw_mandelbrot_set(image, size, width, height, iteration_count); // мб здесь true?
    gdImageTrueColorToPalette(image, 1, 256);
    gdImageGifAnimBegin(image, fout, 1, 0); // добавляем первый кадр
    for(int i = 0; i < num_of_zooms; ++i) {
        gdImageDestroy(image); // удаляем предыдущую картинку, чтобы не было memory leaks
        std::cout << "Current progress: " << 100 * i /num_of_zooms << "%" << std::endl;
        size[0] = x - zoom_coefficient * (x - size[0]);
        size[1] = x + zoom_coefficient * (size[1] - x);
        size[2] = y - zoom_coefficient * (y - size[2]);
        size[3] = y + zoom_coefficient * (size[3] - y);
        draw_mandelbrot_set(image, size, width, height, iteration_count); // отрисовываем новое изображние
        gdImageTrueColorToPalette(image, 1, 256);
        gdImageGifAnimAdd(image, fout, 1, 0, 0, 10, 1, nullptr); // добавляем новый кадр
    }
    gdImageGifAnimEnd(fout); // завершаем gif
    fclose(fout); // закрываем файл
    gdImageDestroy(image); // убираем за собой
}

void draw_single_bmp(std::array<double, 4> size, unsigned int width, unsigned int height, unsigned int iteration_count) {
    FILE* fout;
    gdImagePtr image;
    draw_mandelbrot_set(image, size, width, height, iteration_count);
    fout = fopen("/home/mrgeekman/Документы/Программирование/С++/Программы/Mandelbrot set/Media/Mandelbrot_set.bmp", "wb");
    gdImageBmp(image, fout, 1); // сохраняем картинку
    fclose(fout); // закрываем файл
    gdImageDestroy(image); // убираем за собой
}

void draw_single_gif(std::array<double, 4> size, unsigned int width, unsigned int height, unsigned int iteration_count) {
    FILE* fout;
    gdImagePtr image, image_for_gif;
    draw_mandelbrot_set(image, size, width, height, iteration_count);
    fout = fopen("/home/mrgeekman/Документы/Программирование/С++/Программы/Mandelbrot set/Media/Mandelbrot_set.gif", "wb");
    image_for_gif = gdImageCreatePaletteFromTrueColor(image, 1, 256);
    gdImageGif(image_for_gif, fout); // сохраняем картинку
    fclose(fout); // закрываем файл
    gdImageDestroy(image); // убираем за собой
    gdImageDestroy(image_for_gif);
}

void draw_mandelbrot_set(gdImagePtr& image, std::array<double, 4> size, unsigned int width, unsigned int height, unsigned int iteration_count) {
    unsigned int* matrix = new unsigned int [width * height]; // матрица, представленная в виде массива, куда будет записывать свои результаты CUDA
    compute_matrix(matrix, size[0], size[1], size[2], size[3], width, height, iteration_count);
    // создаем изображение и красим точку
    image = gdImageCreateTrueColor(width, height); // создаем картинку с 24-битным цветом
    for(int row = 0; row < height; ++row) {
        for(int col = 0; col < width; ++col) {
            paint_point(image, col, row, matrix[row * width + col]); // вызываем покраску точки
        }
    }
    // освобождаем память
    delete[] matrix;
}

unsigned int check_point(double x, double y, unsigned int iteration_count) {
    std::complex<double> initial_point(x, y);
    std::complex<double> current_point(0, 0);
    // проверка на принадлежность точки главной картиоиде (можно подумать о включении и выключении)
    double p = sqrt(pow((x - 0.25), 2.0) + pow(y, 2.0));
    double p_c = 0.5 - 0.5 * cos(atan2(y, x - 0.25));
    if(p <= p_c) {
        return 0;
    }
    // проверка точки
    for(unsigned int iteration = 1; iteration <= iteration_count; ++iteration) {
        current_point = pow(current_point, 2) + initial_point;
        if(abs(current_point) >= 2.0) { // что лучше norm() или abs()
            return iteration;
        }
    }
    return 0;
};

void paint_point(gdImagePtr& image, int x, int y, unsigned int iteration) {
    int color = 0;
    if(!iteration) {
        color = gdTrueColor(0, 0, 0);
    } else {
        unsigned int value = iteration % (256 * 7);
        unsigned char stage = value / 256;
        unsigned char step = value % 256;
        switch(stage) {
            case 0:
                color = gdTrueColor(255 - step, 255 - step, 255);
                break;
            case 1:
                color = gdTrueColor(step, 0, 255);
                break;
            case 2:
                color = gdTrueColor(255, 0, 255 - step);
                break;
            case 3:
                color = gdTrueColor(255, step, 0);
                break;
            case 4:
                color = gdTrueColor(255 - step, 255, 0);
                break;
            case 5:
                color = gdTrueColor(0, 255, step);
                break;
            case 6:
                color = gdTrueColor(step, 255, 255);
                break;
        }
    }
    gdImageSetPixel(image, x, y, color);
}

int main() {
    clock_t time = clock();
    std::array<double, 4> size; // лучше сделать с std::array
    size[0] = -2.0; // нижняя граница x
    size[1] = 1.0; // верхняя граница x
    size[2] = -1.25; // нижняя граница y
    size[3] = 1.25; // верхняя граница y
    unsigned int width = 2000; // ширина изображения, которую я хочу получить
    unsigned int height = round(static_cast<double>(width) * (size[3] - size[2]) / (size[1] - size[0])); // высоту посчитаем по пропорции
    unsigned int iteration_count = 2000; // количество итераций для прорисовки
    //draw_zoom_gif(-0.777807810193171, 0.131645108003206, 0.8, 5, width, height, iteration_count);
    draw_single_bmp(size, width, height, iteration_count);
    //draw_single_gif(size, width, height, iteration_count, border);
    time = clock() - time;
    double seconds = static_cast<double>(time)/CLOCKS_PER_SEC;
    int minutes = seconds / 60;
    std::cout << "На выполнение ушло: " << minutes << "m " << seconds - 60 * minutes << "s" << std::endl;
    return 0;
}
