#include <iostream>
#include <cstdio>
#include <array>
#include <complex>
#include <cmath>
#include <ctime>
#include <string>
#include <gd.h>
#include <cuda_runtime_api.h>
#include "opencv2/opencv.hpp"
#include "include/CUDA_wrappers.hpp"

#define PRECISION double

void draw_zoom_video(std::string path, PRECISION x, PRECISION y, double zoom_coefficient, int num_of_frames, unsigned int width, unsigned int height, unsigned int iteration_count);
// рисует видео, которое будет приближаться к определенной точке
void draw_zoom_gif(std::string path, PRECISION x, PRECISION y, double zoom_coefficient, int num_of_zooms, unsigned int width, unsigned int height, unsigned int iteration_count);
// рисует gif, который будет приближаться к определенной точке
void draw_single_bmp(std::string path, std::array<PRECISION, 4> size, unsigned int width, unsigned int height, unsigned int iteration_count);
// рисует одну bmp картинку
void draw_single_gif(std::string path, std::array<PRECISION, 4> size, unsigned int width, unsigned int height, unsigned int iteration_count);
// рисует одну gif картинку
void draw_mandelbrot_set(gdImagePtr& image, std::array<PRECISION, 4> size, unsigned int width, unsigned int height, unsigned int iteration_count);
// рисует множество Мандельброта
void paint_point(gdImagePtr& image, int x, int y, unsigned int iteration);
// красит точку в соответсвии с ее параметрами

void draw_zoom_video(std::string path, PRECISION x, PRECISION y, double zoom_coefficient, int num_of_frames, unsigned int width, unsigned int height, unsigned int iteration_count) {
    unsigned int frames_per_second = 24; // количество кадров в секунду
    path += "media/video/";
    cv::VideoWriter video((path + "Mandelbrot_set_zoom.avi").c_str(), CV_FOURCC('M','J','P','G'), frames_per_second, cv::Size(width, height));
    cv::Mat frame;
    FILE* fout;
    gdImagePtr buffer_image; // изображение, где будет храниться текущий отрисованный кадр
    std::array<PRECISION, 4> size;
    PRECISION add_x = 2.0;
    PRECISION add_y = static_cast<double>(height) * add_x / static_cast<double>(width);
    size[0] = x - add_x; // нижняя граница x
    size[1] = x + add_x; // верхняя граница x
    size[2] = y - add_y; // нижняя граница y
    size[3] = y + add_y; // верхняя граница y
    for(int frame_number = 0; frame_number < num_of_frames; ++frame_number) {
        // создаем временное изображение и сохраняем его
        fout = fopen((path + "Video_buffer.bmp").c_str(), "wb");
        draw_mandelbrot_set(buffer_image, size, width, height, iteration_count); // рисуем изображение и помещаем его в наш буфер
        gdImageBmp(buffer_image, fout, 1); // сохраняем картинку
        fclose(fout); // закрываем файл
        gdImageDestroy(buffer_image); // убираем за собой
        // открываем изображение и добавляем его в видео
        frame = cv::imread((path + "Video_buffer.bmp").c_str());
        video.write(frame);
        // пересчитываем параметры изображения для следующего кадра
        size[0] = x - zoom_coefficient * (x - size[0]);
        size[1] = x + zoom_coefficient * (size[1] - x);
        size[2] = y - zoom_coefficient * (y - size[2]);
        size[3] = y + zoom_coefficient * (size[3] - y);
        // выводим текущий прогресс в создании видео
        std::cout << "Current progress: " << 100 * frame_number /num_of_frames << "%" << std::endl;
    }
    video.release();
    cv::destroyAllWindows();
}

void draw_zoom_gif(std::string path, PRECISION x, PRECISION y, double zoom_coefficient, int num_of_zooms, unsigned int width, unsigned int height, unsigned int iteration_count) {
    path += "media/";
    gdImagePtr image;
    FILE* fout;
    fout = fopen((path + "Mandelbrot_set_zoom.gif").c_str(), "wb");
    std::array<PRECISION, 4> size;
    PRECISION add_x = 2.0;
    PRECISION add_y = static_cast<double>(height) * add_x / static_cast<double>(width);
    size[0] = x - add_x; // нижняя граница x
    size[1] = x + add_x; // верхняя граница x
    size[2] = y - add_y; // нижняя граница y
    size[3] = y + add_y; // верхняя граница y
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
    // убираем за собой
    fclose(fout);
    gdImageDestroy(image);
}

void draw_single_bmp(std::string path, std::array<PRECISION, 4> size, unsigned int width, unsigned int height, unsigned int iteration_count) {
    path += "media/";
    FILE* fout;
    gdImagePtr image;
    draw_mandelbrot_set(image, size, width, height, iteration_count);
    fout = fopen((path + "Mandelbrot_set.bmp").c_str(), "wb");
    gdImageBmp(image, fout, 1); // сохраняем картинку
    // убираем за собой
    fclose(fout);
    gdImageDestroy(image);
}

void draw_single_gif(std::string path, std::array<PRECISION, 4> size, unsigned int width, unsigned int height, unsigned int iteration_count) {
    path += "media/";
    FILE* fout;
    gdImagePtr image, image_for_gif;
    draw_mandelbrot_set(image, size, width, height, iteration_count);
    fout = fopen((path + "Mandelbrot_set.gif").c_str(), "wb");
    image_for_gif = gdImageCreatePaletteFromTrueColor(image, 1, 256);
    gdImageGif(image_for_gif, fout); // сохраняем картинку
    // убираем за собой
    fclose(fout);
    gdImageDestroy(image);
    gdImageDestroy(image_for_gif);
}

void draw_mandelbrot_set(gdImagePtr& image, std::array<PRECISION, 4> size, unsigned int width, unsigned int height, unsigned int iteration_count) {
    unsigned int* matrix = new unsigned int [width * height]; // матрица, представленная в виде массива, куда будет записывать свои результаты CUDA
    compute_matrix(matrix, size[0], size[1], size[2], size[3], width, height, iteration_count);
    // создаем изображение и красим точку
    image = gdImageCreateTrueColor(width, height); // создаем картинку с 24-битным цветом
    for(unsigned int row = 0; row < height; ++row) {
        for(unsigned int col = 0; col < width; ++col) {
            paint_point(image, col, row, matrix[row * width + col]); // вызываем покраску точки
        }
    }
    // освобождаем память
    delete[] matrix;
}

void paint_point(gdImagePtr& image, int x, int y, unsigned int iteration) {
    int color = 0;
    if(!iteration) {
        color = gdTrueColor(0, 0, 0);
    } else {
        iteration *= 5; // влияет на резкость цветовых переходов
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
    std::string path = "/home/mrgeekman/Документы/Программирование/С++/Программы/Mandelbrot set/"; // текущая директория
    std::array<PRECISION, 4> size;
    size[0] = -2.0; // нижняя граница x
    size[1] = 2.0; // верхняя граница x
    size[2] = -1.125; // нижняя граница y
    size[3] = 1.125; // верхняя граница y
    unsigned int width = 960; // ширина изображения
    unsigned int height = round(static_cast<PRECISION>(width) * (size[3] - size[2]) / (size[1] - size[0])); // высота изображения
    unsigned int iteration_count = 5000; // количество итераций для прорисовки
    // Здесь необходимо выбрать то, что хотите отрисовать
    time = clock() - time;
    double seconds = static_cast<double>(time)/CLOCKS_PER_SEC;
    int minutes = seconds / 60;
    std::cout << "На выполнение ушло: " << minutes << "m " << seconds - 60 * minutes << "s" << std::endl;
    return 0;
}
