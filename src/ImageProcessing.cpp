#include "ImageProcessing.h"
#include <algorithm>
#include <QDebug>

//  QImage -> double
std::vector<std::vector<double>> ImageProcessing::imageToDouble(const QImage& image) {
    int width = image.width();
    int height = image.height();

    std::vector<std::vector<double>> imageData(height, std::vector<double>(width, 0.0));
   
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            int gray = qGray(image.pixel(y, x));  // qGray vracia hodnotu v stupòoch šedej
            imageData[x][y] = gray / 255.0;  // Normalizácia na rozsah 0-1
        }
    }

    return imageData;
}

//  double -> QImage
QImage ImageProcessing::doubleToImage(const std::vector<std::vector<double>>& imageData, int width, int height) {
    QImage result(width, height, QImage::Format_Grayscale8);

    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            int gray = std::clamp(static_cast<int>(imageData[x][y] * 255.0), 0, 255);
            //clamp, hodnoty ostanu v rozsahu 0-255, static_cast<int> - konverzia na cele cislo
            result.setPixelColor(y, x, QColor(gray, gray, gray));
        }
    }

    return result;
}

//  Inverzia farieb
QImage ImageProcessing::invertColors(const QImage& image) {
    auto imageData = imageToDouble(image);//auto odhadne typ premenej na zaklade priradenej hodnoty
    int width = image.width();
    int height = image.height();

    for (int x = 0; x < height; x++) {  
        for (int y = 0; y < width; y++) {  
            imageData[x][y] = 1.0 - imageData[x][y];  // Inverzia: 1.0 - hodnota
        }
    }

    return doubleToImage(imageData, image.width(), image.height());
}

QImage ImageProcessing::FSHS(const QImage& image)
{
    auto imageData = imageToDouble(image);
    double min_val = 1.0, max_val = 0.0;

    int width = image.width();
    int height = image.height();

    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            double val = imageData[x][y];  
            //najdenie minima a maxima
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
    }

    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            imageData[x][y] = (imageData[x][y] - min_val) / (max_val - min_val);  // Roztiahnutie histogramu
        }
    }

    return doubleToImage(imageData, image.width(), image.height());
}

QImage ImageProcessing::EH(const QImage& image)
{
    int width = image.width();
    int height = image.height();
    int numPixels = width * height;
    const int histSize = 256;

    std::vector<int> histogram(histSize, 0);//vytvorenie histogramu

    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            QColor pixel = image.pixelColor(y, x);
            int intensity = pixel.red();
            histogram[intensity]++; //pocet pixlov s danou intenzitou
        }
    }

    std::vector<double> cdf(histSize, 0);//komultativna distribucna funkcia
    cdf[0] = histogram[0];

    for (int i = 1; i < histSize; i++) {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    for (int i = 0; i < histSize; i++) {
        cdf[i] /= numPixels;//normalizacia na 0-1
    }

    // Aplikácia ekvalizácie
    QImage result = image.copy(); 
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            QColor pixel = result.pixelColor(y, x);
            int intensity = pixel.red();
            int newIntensity = static_cast<int>(cdf[intensity] * 255); // Prevod na nový intenzitu
            result.setPixelColor(y, x, QColor(newIntensity, newIntensity, newIntensity));
        }
    }

    return result;
}

QImage ImageProcessing::mirrorImage(const QImage& image, int d)
{
    int width = image.width();
    int height = image.height();

    int new_width = width + (d * 2);
    int new_height = height + (d * 2);

    QImage mirroredImage(new_width, new_height, QImage::Format_Grayscale8);

    //prekreslenie stareho obrazka
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            int gray = qGray(image.pixel(y, x));
            mirroredImage.setPixel(y + d, x + d, qRgb(gray, gray, gray));
        }
    }

    // Horný a dolný okraj
    for (int y = 0; y < width; y++) {
        for (int i = 0; i < d; i++) {
            int grayTop = qGray(image.pixel(y, i));                          // Horný okraj
            int grayBottom = qGray(image.pixel(y, height - 1 - i));      // Dolný okraj
            mirroredImage.setPixel(y + d, d - 1 - i, qRgb(grayTop, grayTop, grayTop));
            mirroredImage.setPixel(y + d, height + d + i, qRgb(grayBottom, grayBottom, grayBottom));
        }
    }

    // ¼avý a pravý okraj
    for (int x = 0; x < new_height; x++) {
        for (int i = 0; i < d; i++) {
            QRgb leftPixel = mirroredImage.pixel(i + d, x);   // V¾avo
            QRgb rightPixel = mirroredImage.pixel(width + d - 1 - i, x);  // Vpravo
            mirroredImage.setPixel(d - 1 - i, x, leftPixel);
            mirroredImage.setPixel(width + d + i, x, rightPixel);
        }
    }
    mirroredImage.save("mirrored.jpg");
    
    return mirroredImage;
}

std::vector<std::vector<double>> ImageProcessing::mirrorImage(const std::vector<std::vector<double>>& image, int d)
{
    int width = image[0].size();
    int height = image.size();

    int new_width = width + (d * 2);
    int new_height = height + (d * 2);

    std::vector<std::vector<double>> mirroredImage(new_height, std::vector<double>(new_width, 0.0));

    //prekreslenie stareho obrazka
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            mirroredImage[x + d][y + d] = image[x][y];
        }
    }

    // Horný a dolný okraj
    for (int y = 0; y < width; y++) {
        for (int i = 0; i < d; i++) {
            mirroredImage[d - 1 - i][y + d] = image[i][y];                 // Horný okraj
            mirroredImage[height + d + i][y + d] = image[height - 1 - i][y]; // Dolný okraj
        }
    }

    // ¼avý a pravý okraj
    for (int x = 0; x < new_height; x++) {
        for (int i = 0; i < d; i++) {
            mirroredImage[x][d - 1 - i] = mirroredImage[x][i + d];            // ¼avý okraj
            mirroredImage[x][width + d + i] = mirroredImage[x][width + d - 1 - i]; // Pravý okraj
        }
    }

    return mirroredImage;
}

QImage ImageProcessing::linearConvolution(const QImage& image)
{
    int width = image.width();
    int height = image.height();
    int d = 2;

    auto mirroredData = imageToDouble(mirrorImage(image, d));

    //vytvorenie masky
    std::vector<std::vector<double>> mask = {
    {0.000077693991227, 0.001813519368126, 0.005031312077870, 0.001813519368126, 0.000077693991227},
    {0.001813519368126, 0.042330847554975, 0.117439994473243, 0.042330847554975, 0.001813519368126},
    {0.005031312077870, 0.117439994473243, 0.325972452665734, 0.117439994473243, 0.005031312077870},
    {0.001813519368126, 0.042330847554975, 0.117439994473243, 0.042330847554975, 0.001813519368126},
    {0.000077693991227, 0.001813519368126, 0.005031312077870, 0.001813519368126, 0.000077693991227}
    };
  
    std::vector<std::vector<double>> outputData(height, std::vector<double>(width, 0.0));

    // konvolucia
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            double sum = 0;

            for (int i = -d; i <= d; i++) {
                for (int j = -d; j <= d; j++) {// prechádzanie masky
                    int pixelX = x + j + d;//urcenie pixla zrkadleneho obrazka
                    int pixelY = y + i + d;

                    double pixelVal = mirroredData[pixelX][pixelY];//hodnota z pixla zrkadleneho obraza
                    double weight = mask[i + d][j + d];//vaha z masky

                    sum += pixelVal * weight;
                }
            }

            outputData[x][y] = std::clamp(sum, 0.0, 1.0); 
        }
    }
    qDebug() << "konvolucia";

    return doubleToImage(outputData, width, height);
}

QImage ImageProcessing::explicit_scheme(const QImage& image, int T, double tau)
{
    int width = image.width();
    int height = image.height();
    int d = 1;
    double up, down, left, right;
    double num_pix = width * height;

    auto imageData = imageToDouble(image);

    double avg_sum = 0.0;
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            avg_sum += imageData[x][y];
        }
    }
    double avg = avg_sum / num_pix;

    auto mirroredData = imageToDouble(mirrorImage(image, d));
    std::vector<std::vector<double>> outputData(height, std::vector<double>(width, 0.0));

    for (int t = 0; t < T; t++) {
        double avg_new_sum = 0, avg_new;
        if (t == 0) {
            for (int x = 0; x < height; x++) {
                for (int y = 0; y < width; y++) {
                    double sum = 0.0;

                    up = mirroredData[x][y + 1];
                    down = mirroredData[x + 2][y + 1];
                    left = mirroredData[x + 1][y];
                    right = mirroredData[x + 1][y + 2];

                    sum = up + down + left + right;
                    
                    outputData[x][y] = (1 - 4 * tau) * imageData[x][y] + tau * sum;

                    avg_new_sum += outputData[x][y];
                }
            }
        }
        else
        {
            mirroredData = mirrorImage(outputData, d);
            for (int x = 0; x < height; x++) {
                for (int y = 0; y < width; y++) {
                    double sum = 0.0;

                    up = mirroredData[x][y + 1];
                    down = mirroredData[x + 2][y + 1];
                    left = mirroredData[x + 1][y];
                    right = mirroredData[x + 1][y + 2];

                    sum = up + down + left + right;

                    outputData[x][y] = (1 - 4 * tau) * mirroredData[x + 1][y + 1] + tau * sum;

                    avg_new_sum += outputData[x][y];
                }
            }
        }
        avg_new = avg_new_sum / num_pix;
        qDebug() << "casovy krok:" << t << "povodny priemer:" << avg << "novy priemer:" << avg_new;
    }
    return doubleToImage(outputData, width, height);
}

QImage ImageProcessing::implicit_scheme(const QImage& image, int T, double tau)
{
    int width = image.width();
    int height = image.height();
    int d = 1; 
    int maxIter = 100;
    double omega = 1.25;
    double tol = 0.00001;
    double num_pix = width * height;

    auto imageData = imageToDouble(image);

    double avg_sum = 0.0;
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            avg_sum += imageData[x][y];
        }
    }
    double avg = avg_sum / num_pix;

    auto outputData = imageData;

    for (int t = 0; t < T; t++) {
        auto newData = outputData;
        auto mirroredData = mirrorImage(newData, d);
        int count = 0;
        double resSum;

        for (int i = 0; i < maxIter; i++) {
            resSum = 0.0;
            for (int x = 0; x < height; x++) {
                for (int y = 0; y < width; y++) {
                    double oldVal = newData[x][y];

                    double sum = mirroredData[x][y + 1] + mirroredData[x + 1][y] +
                        mirroredData[x + 1][y + 2] + mirroredData[x + 2][y + 1];

                    newData[x][y] = (1 - omega) * oldVal + omega * ((outputData[x][y] + tau * sum) / (1 + 4 * tau));

                    double residual = outputData[x][y] - ((1 + 4 * tau) * newData[x][y] - tau * sum);                    
                    resSum += residual * residual;                     
                }
            }
            count++;
            if (sqrt(resSum) < tol) break;

        }

        double avg_new_sum = 0.0;
        for (int x = 0; x < height; x++) {
            for (int y = 0; y < width; y++) {
                avg_new_sum += newData[x][y];
            }
        }
        double avg_new = avg_new_sum / num_pix;

        qDebug() << "casovy krok:" << t << "povodny priemer:" << avg << "novy priemer:" << avg_new << "SOR iteracie:" << count << "reziduum:" << sqrt(resSum);

        outputData = newData;
    }

    return doubleToImage(outputData, width, height);
}

QImage ImageProcessing::EdgesDetection(const QImage& image, int K)
{
    int width = image.width();
    int height = image.height();
    int d = 1;
    int h = 1;
    auto imageData = imageToDouble(image);
    auto mirroredData = mirrorImage(imageData, d);
    auto outputData = imageData;

    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            
            double gradient_east_x = (mirroredData[x + 1][y + 2] - mirroredData[x + 1][y + 1]) / h;
            double gradient_east_y = (mirroredData[x][y + 2] - mirroredData[x + 2][y + 2] + 
                mirroredData[x][y + 1] - mirroredData[x + 2][y + 1]) / (4 * h);
            double gradient_east = sqrt(pow(gradient_east_x, 2) + pow(gradient_east_y, 2));

            double gradient_north_x = (mirroredData[x][y + 1] - mirroredData[x + 1][y + 1]) / h;
            double gradient_north_y = (mirroredData[x][y] - mirroredData[x][y + 2] + 
                mirroredData[x + 1][y] - mirroredData[x + 1][y + 2]) / (4 * h);
            double gradient_north = sqrt(pow(gradient_north_x, 2) + pow(gradient_north_y, 2));

            double gradient_west_x = (mirroredData[x + 1][y] - mirroredData[x + 1][y + 1]) / h;
            double gradient_west_y = (mirroredData[x][y] - mirroredData[x + 2][y] +
                mirroredData[x][y + 1] - mirroredData[x + 2][y + 1]) / (4 * h);
            double gradient_west = sqrt(pow(gradient_west_x, 2) + pow(gradient_west_y, 2));

            double gradient_south_x = (mirroredData[x + 2][y + 1] - mirroredData[x + 1][y + 1]) / h;
            double gradient_south_y = (mirroredData[x + 2][y] - mirroredData[x + 2][y + 2] +
                mirroredData[x + 1][y] - mirroredData[x + 1][y + 2]) / (4 * h);
            double gradient_south = sqrt(pow(gradient_south_x, 2) + pow(gradient_south_y, 2));

            double gradient_avg = (gradient_east + gradient_north + gradient_west + gradient_south) / 4;

            outputData[x][y] = 1 / (1 + K * pow(gradient_avg, 2));
        }
    }

    return doubleToImage(outputData, width, height);
}

QImage ImageProcessing::PeroneMalik(const QImage& image, int T, double tau, double sigma, int K)
{
    int width = image.width();
    int height = image.height();
    int d = 1;
    int maxIter = 100;
    double omega = 1.25;
    double tol = 0.00001;
    int h = 1;
    std::vector<std::vector<double>> g_n(height, std::vector<double>(width, 0.0));
    std::vector<std::vector<double>> g_w(height, std::vector<double>(width, 0.0));
    std::vector<std::vector<double>> g_s(height, std::vector<double>(width, 0.0));
    std::vector<std::vector<double>> g_e(height, std::vector<double>(width, 0.0));

    auto imageData = imageToDouble(image);

    double num_pix = width * height;

    double avg_sum = 0.0;
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            avg_sum += imageData[x][y];
        }
    }
    double avg = avg_sum / num_pix;

    auto outputData = imageData;

    for (int t = 0; t < T; t++) {
        auto newData = outputData;
        auto mirroredData = mirrorImage(newData, d);
        int count = 0;
        double resSum = 0.0;

        if (sigma > 0.25) {
            for (int i = 0; i < maxIter; i++) {
                resSum = 0.0;
                for (int x = 0; x < height; x++) {
                    for (int y = 0; y < width; y++) {
                        double oldVal = newData[x][y];

                        double sum = mirroredData[x][y + 1] + mirroredData[x + 1][y] +
                            mirroredData[x + 1][y + 2] + mirroredData[x + 2][y + 1];

                        newData[x][y] = (1 - omega) * oldVal + omega * ((outputData[x][y] + sigma * sum) / (1 + 4 * sigma));
                        double residual = outputData[x][y] - ((1 + 4 * sigma) * newData[x][y] - sigma * sum);
                        resSum += residual * residual;
                    }
                }
                if (sqrt(resSum) < tol) break;
            }
        }
        else {
            for (int x = 0; x < height; x++) {
                for (int y = 0; y < width; y++) {
                    double sum = 0.0;

                    double up = mirroredData[x][y + 1];
                    double down = mirroredData[x + 2][y + 1];
                    double left = mirroredData[x + 1][y];
                    double right = mirroredData[x + 1][y + 2];

                    sum = up + down + left + right;

                    newData[x][y] = (1 - 4 * sigma) * mirroredData[x + 1][y + 1] + sigma * sum;

                }
            }
        }

        mirroredData = mirrorImage(newData, d);
        for (int x = 0; x < height; x++) {
            for (int y = 0; y < width; y++) {
                double gradient_east_x = (mirroredData[x + 1][y + 2] - mirroredData[x + 1][y + 1]) / h;
                double gradient_east_y = (mirroredData[x][y + 2] - mirroredData[x + 2][y + 2] +
                    mirroredData[x][y + 1] - mirroredData[x + 2][y + 1]) / (4 * h);
                g_e[x][y] = 1 / (1 + K * (pow(gradient_east_x, 2) + pow(gradient_east_y, 2)));

                double gradient_north_x = (mirroredData[x][y + 1] - mirroredData[x + 1][y + 1]) / h;
                double gradient_north_y = (mirroredData[x][y] - mirroredData[x][y + 2] +
                    mirroredData[x + 1][y] - mirroredData[x + 1][y + 2]) / (4 * h);
                g_n[x][y] = 1 / (1 + K * (pow(gradient_north_x, 2) + pow(gradient_north_y, 2)));

                double gradient_west_x = (mirroredData[x + 1][y] - mirroredData[x + 1][y + 1]) / h;
                double gradient_west_y = (mirroredData[x][y] - mirroredData[x + 2][y] +
                    mirroredData[x][y + 1] - mirroredData[x + 2][y + 1]) / (4 * h);
                g_w[x][y] = 1 / (1 + K * (pow(gradient_west_x, 2) + pow(gradient_west_y, 2)));

                double gradient_south_x = (mirroredData[x + 2][y + 1] - mirroredData[x + 1][y + 1]) / h;
                double gradient_south_y = (mirroredData[x + 2][y] - mirroredData[x + 2][y + 2] +
                    mirroredData[x + 1][y] - mirroredData[x + 1][y + 2]) / (4 * h);
                g_s[x][y] = 1/(1 + K*(pow(gradient_south_x, 2) + pow(gradient_south_y, 2)));

            }
        }

        for (int i = 0; i < maxIter; i++) {
            resSum = 0.0;

            for (int x = 0; x < height; x++) {
                for (int y = 0; y < width; y++) {
                    double oldVal = newData[x][y];

                    double sum = mirroredData[x][y + 1] * g_n[x][y] + mirroredData[x + 1][y] * g_w[x][y] +
                        mirroredData[x + 1][y + 2] * g_e[x][y] + mirroredData[x + 2][y + 1] * g_s[x][y];

                    double g_sum = g_n[x][y] + g_s[x][y] + g_w[x][y] + g_e[x][y];

                    newData[x][y] = (1 - omega) * oldVal +
                        omega * ((outputData[x][y] + tau * sum) / (1 + tau * g_sum));

                    double residual = outputData[x][y] - ((1 + tau * g_sum) * newData[x][y] - tau * sum);
                    resSum += residual * residual;
                }
            }
            count++;
            if (sqrt(resSum) < tol) break;

        }

        double avg_new_sum = 0.0;
        for (int x = 0; x < height; x++) {
            for (int y = 0; y < width; y++) {
                avg_new_sum += newData[x][y];
            }
        }
        double avg_new = avg_new_sum / num_pix;

        qDebug() << "casovy krok:" << t << "povodny priemer:" << avg << "novy priemer:" << avg_new << 
            "SOR iteracie:" << count << "reziduum:" << sqrt(resSum);

        outputData = newData;
    }

    return doubleToImage(outputData, width, height);
}

QImage ImageProcessing::MCF(const QImage& image, int T, double tau, double epsilon)
{
    int width = image.width();
    int height = image.height();
    int d = 1;
    int maxIter = 100;
    double omega = 1.25;
    double tol = 0.00001;
    int h = 1;
    std::vector<std::vector<double>> g_n(height, std::vector<double>(width, 0.0));
    std::vector<std::vector<double>> g_w(height, std::vector<double>(width, 0.0));
    std::vector<std::vector<double>> g_s(height, std::vector<double>(width, 0.0));
    std::vector<std::vector<double>> g_e(height, std::vector<double>(width, 0.0));
    std::vector<std::vector<double>> avg_grad_eps(height, std::vector<double>(width, 0.0));

    auto imageData = imageToDouble(image);

    auto outputData = imageData;

    for (int t = 0; t < T; t++) {
        auto newData = outputData;
        auto mirroredData = mirrorImage(newData, d);
        int count = 0;
        double resSum = 0.0;
        
        for (int x = 0; x < height; x++) {
            for (int y = 0; y < width; y++) {
                double gradient_east_x = (mirroredData[x + 1][y + 2] - mirroredData[x + 1][y + 1]) / h;
                double gradient_east_y = (mirroredData[x][y + 2] - mirroredData[x + 2][y + 2] +
                    mirroredData[x][y + 1] - mirroredData[x + 2][y + 1]) / (4 * h);
                double grad_east_eps = sqrt(pow(gradient_east_x, 2) + pow(gradient_east_y, 2) + epsilon * epsilon);
                g_e[x][y] = 1 / grad_east_eps;

                double gradient_north_x = (mirroredData[x][y + 1] - mirroredData[x + 1][y + 1]) / h;
                double gradient_north_y = (mirroredData[x][y] - mirroredData[x][y + 2] +
                    mirroredData[x + 1][y] - mirroredData[x + 1][y + 2]) / (4 * h);
                double grad_north_eps = sqrt(pow(gradient_north_x, 2) + pow(gradient_north_y, 2) + epsilon * epsilon);
                g_n[x][y] = 1 / grad_north_eps;

                double gradient_west_x = (mirroredData[x + 1][y] - mirroredData[x + 1][y + 1]) / h;
                double gradient_west_y = (mirroredData[x][y] - mirroredData[x + 2][y] +
                    mirroredData[x][y + 1] - mirroredData[x + 2][y + 1]) / (4 * h);
                double grad_west_eps = sqrt(pow(gradient_west_x, 2) + pow(gradient_west_y, 2) + epsilon * epsilon);
                g_w[x][y] = 1 / grad_west_eps;

                double gradient_south_x = (mirroredData[x + 2][y + 1] - mirroredData[x + 1][y + 1]) / h;
                double gradient_south_y = (mirroredData[x + 2][y] - mirroredData[x + 2][y + 2] +
                    mirroredData[x + 1][y] - mirroredData[x + 1][y + 2]) / (4 * h);
                double grad_south_eps = sqrt(pow(gradient_south_x, 2) + pow(gradient_south_y, 2) + epsilon * epsilon);
                g_s[x][y] = 1 / grad_south_eps;

                avg_grad_eps[x][y] = (grad_north_eps + grad_east_eps + grad_west_eps + grad_south_eps) / 4;
            }
        }

        for (int i = 0; i < maxIter; i++) {
            resSum = 0.0;

            for (int x = 0; x < height; x++) {
                for (int y = 0; y < width; y++) {
                    double oldVal = newData[x][y];

                    double sum = mirroredData[x][y + 1] * g_n[x][y] + mirroredData[x + 1][y] * g_w[x][y] +
                        mirroredData[x + 1][y + 2] * g_e[x][y] + mirroredData[x + 2][y + 1] * g_s[x][y];

                    double g_sum = g_n[x][y] + g_s[x][y] + g_w[x][y] + g_e[x][y];

                    newData[x][y] = (1 - omega) * oldVal +
                        omega * ((outputData[x][y] + tau * avg_grad_eps[x][y] * sum) / (1 + tau * avg_grad_eps[x][y] * g_sum));

                    double residual = outputData[x][y] - ((1 + tau * avg_grad_eps[x][y] * g_sum) * newData[x][y] - tau * avg_grad_eps[x][y] * sum);
                    resSum += residual * residual;
                }
            }
            count++;
            if (sqrt(resSum) < tol) break;

        }

        qDebug() << "casovy krok:" << t << "SOR iteracie:" << count << "reziduum:" << sqrt(resSum);

        outputData = newData;
    }

    return doubleToImage(outputData, width, height);
}

QImage ImageProcessing::GMCF(const QImage& image, int T, double tau, double sigma, int K, double epsilon)
{
    int width = image.width();
    int height = image.height();
    int d = 1;
    int maxIter = 100;
    double omega = 1.25;
    double tol = 0.00001;
    int h = 1;

    std::vector<std::vector<double>> g_n(height, std::vector<double>(width, 0.0));
    std::vector<std::vector<double>> g_w(height, std::vector<double>(width, 0.0));
    std::vector<std::vector<double>> g_s(height, std::vector<double>(width, 0.0));
    std::vector<std::vector<double>> g_e(height, std::vector<double>(width, 0.0));
    std::vector<std::vector<double>> avg_grad_eps(height, std::vector<double>(width, 0.0));
    std::vector<std::vector<double>> grad_south_eps(height, std::vector<double>(width, 0.0));
    std::vector<std::vector<double>> grad_north_eps(height, std::vector<double>(width, 0.0));
    std::vector<std::vector<double>> grad_west_eps(height, std::vector<double>(width, 0.0));
    std::vector<std::vector<double>> grad_east_eps(height, std::vector<double>(width, 0.0));


    auto imageData = imageToDouble(image);

    double num_pix = width * height;

    auto outputData = imageData;

    for (int t = 0; t < T; t++) {
        auto newData = outputData;
        auto mirroredData = mirrorImage(newData, d);
        
        int count = 0;
        double resSum = 0.0;

        for (int x = 0; x < height; x++) {
            for (int y = 0; y < width; y++) {
                double gradient_east_x = (mirroredData[x + 1][y + 2] - mirroredData[x + 1][y + 1]) / h;
                double gradient_east_y = (mirroredData[x][y + 2] - mirroredData[x + 2][y + 2] +
                    mirroredData[x][y + 1] - mirroredData[x + 2][y + 1]) / (4 * h);
                grad_east_eps[x][y] = sqrt(pow(gradient_east_x, 2) + pow(gradient_east_y, 2) + epsilon * epsilon);


                double gradient_north_x = (mirroredData[x][y + 1] - mirroredData[x + 1][y + 1]) / h;
                double gradient_north_y = (mirroredData[x][y] - mirroredData[x][y + 2] +
                    mirroredData[x + 1][y] - mirroredData[x + 1][y + 2]) / (4 * h);
                grad_north_eps[x][y] = sqrt(pow(gradient_north_x, 2) + pow(gradient_north_y, 2) + epsilon * epsilon);

                double gradient_west_x = (mirroredData[x + 1][y] - mirroredData[x + 1][y + 1]) / h;
                double gradient_west_y = (mirroredData[x][y] - mirroredData[x + 2][y] +
                    mirroredData[x][y + 1] - mirroredData[x + 2][y + 1]) / (4 * h);
                grad_west_eps[x][y] = sqrt(pow(gradient_west_x, 2) + pow(gradient_west_y, 2) + epsilon * epsilon);

                double gradient_south_x = (mirroredData[x + 2][y + 1] - mirroredData[x + 1][y + 1]) / h;
                double gradient_south_y = (mirroredData[x + 2][y] - mirroredData[x + 2][y + 2] +
                    mirroredData[x + 1][y] - mirroredData[x + 1][y + 2]) / (4 * h);
                grad_south_eps[x][y] = sqrt(pow(gradient_south_x, 2) + pow(gradient_south_y, 2) + epsilon * epsilon);

                avg_grad_eps[x][y] = (grad_north_eps[x][y] + grad_east_eps[x][y] + grad_west_eps[x][y] + grad_south_eps[x][y]) / 4;
            }
        }

        if (sigma > 0.25) {
            for (int i = 0; i < maxIter; i++) {
                resSum = 0.0;
                for (int x = 0; x < height; x++) {
                    for (int y = 0; y < width; y++) {
                        double oldVal = newData[x][y];

                        double sum = mirroredData[x][y + 1] + mirroredData[x + 1][y] +
                            mirroredData[x + 1][y + 2] + mirroredData[x + 2][y + 1];

                        newData[x][y] = (1 - omega) * oldVal + omega * ((outputData[x][y] + sigma * sum) / (1 + 4 * sigma));

                        double residual = outputData[x][y] - ((1 + 4 * sigma) * newData[x][y] - sigma * sum);
                        resSum += residual * residual;

                    }
                }
                if (sqrt(resSum) < tol) break;
            }
        }
        else {
            for (int x = 0; x < height; x++) {
                for (int y = 0; y < width; y++) {
                    double sum = 0.0;

                    double up = mirroredData[x][y + 1];
                    double down = mirroredData[x + 2][y + 1];
                    double left = mirroredData[x + 1][y];
                    double right = mirroredData[x + 1][y + 2];

                    sum = up + down + left + right;

                    newData[x][y] = (1 - 4 * sigma) * mirroredData[x + 1][y + 1] + sigma * sum;

                }
            }
        }

        mirroredData = mirrorImage(newData, d);
        for (int x = 0; x < height; x++) {
            for (int y = 0; y < width; y++) {
                double gradient_east_x = (mirroredData[x + 1][y + 2] - mirroredData[x + 1][y + 1]) / h;
                double gradient_east_y = (mirroredData[x][y + 2] - mirroredData[x + 2][y + 2] +
                    mirroredData[x][y + 1] - mirroredData[x + 2][y + 1]) / (4 * h);
                g_e[x][y] = (1 / (1 + K * (pow(gradient_east_x, 2) + pow(gradient_east_y, 2)))) * (1 / grad_east_eps[x][y]);

                double gradient_north_x = (mirroredData[x][y + 1] - mirroredData[x + 1][y + 1]) / h;
                double gradient_north_y = (mirroredData[x][y] - mirroredData[x][y + 2] +
                    mirroredData[x + 1][y] - mirroredData[x + 1][y + 2]) / (4 * h);
                g_n[x][y] = (1 / (1 + K * (pow(gradient_north_x, 2) + pow(gradient_north_y, 2)))) * (1 / grad_north_eps[x][y]);

                double gradient_west_x = (mirroredData[x + 1][y] - mirroredData[x + 1][y + 1]) / h;
                double gradient_west_y = (mirroredData[x][y] - mirroredData[x + 2][y] +
                    mirroredData[x][y + 1] - mirroredData[x + 2][y + 1]) / (4 * h);
                g_w[x][y] = (1 / (1 + K * (pow(gradient_west_x, 2) + pow(gradient_west_y, 2)))) * (1 / grad_west_eps[x][y]);

                double gradient_south_x = (mirroredData[x + 2][y + 1] - mirroredData[x + 1][y + 1]) / h;
                double gradient_south_y = (mirroredData[x + 2][y] - mirroredData[x + 2][y + 2] +
                    mirroredData[x + 1][y] - mirroredData[x + 1][y + 2]) / (4 * h);
                g_s[x][y] = (1 / (1 + K * (pow(gradient_south_x, 2) + pow(gradient_south_y, 2)))) * (1 / grad_south_eps[x][y]);
            }
        }

        for (int i = 0; i < maxIter; i++) {
            resSum = 0.0;

            for (int x = 0; x < height; x++) {
                for (int y = 0; y < width; y++) {
                    double oldVal = newData[x][y];

                    double sum = mirroredData[x][y + 1] * g_n[x][y]  + mirroredData[x + 1][y] * g_w[x][y] +
                        mirroredData[x + 1][y + 2] * g_e[x][y] + mirroredData[x + 2][y + 1] * g_s[x][y];

                    double g_sum = g_n[x][y] + g_s[x][y] + g_w[x][y] + g_e[x][y];
                    newData[x][y] = (1 - omega) * oldVal +
                        omega * ((outputData[x][y] + tau * avg_grad_eps[x][y] * sum) / (1 + tau * g_sum * avg_grad_eps[x][y]));

                    double residual = outputData[x][y] - ((1 + tau * g_sum * avg_grad_eps[x][y]) * newData[x][y] - tau * avg_grad_eps[x][y] * sum);
                    resSum += residual * residual;
                }
            }
            count++;
            if (sqrt(resSum) < tol) break;

        }

        qDebug() << "casovy krok:" << t << "SOR iteracie:" << count << "reziduum:" << sqrt(resSum);

        outputData = newData;
    }

    return doubleToImage(outputData, width, height);
}

std::vector<std::vector<double>> ImageProcessing::normalizeDoubleMatrix(const std::vector<std::vector<double>>& image) {
    double minVal = 10000000.0;
    double maxVal = -1000000.0;

    int height = image.size();
    int width = image[0].size();

    // Najprv nájdeme minimum a maximum v matici
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            if (image[x][y] < minVal) {
                minVal = image[x][y];
            }
            if (image[x][y] > maxVal) {
                maxVal = image[x][y];
            }
        }
    }

    std::vector<std::vector<double>> normalized(height, std::vector<double>(width, 0.0));

    double range = maxVal - minVal;
    if (range == 0.0) {
        range = 1.0;
    }

    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            normalized[x][y] = (image[x][y] - minVal) / range;
        }
    }

    return normalized;
}

QImage ImageProcessing::Rouy_Tourin(const QImage& image, double tau, int K)
{
    int width = image.width();
    int height = image.height();
    int d = 1;
    int h = 1;
    double eps = 0.000001;
    int maxIter = 1000;

    std::vector<std::vector<int>> F(height, std::vector<int>(width, 1));
    std::vector<std::vector<double>> dist(height + d * 2, std::vector<double>(width + d * 2, 0.0));

    auto imageData = imageToDouble(image);
    auto outputData = imageData;

    auto mirroredData = mirrorImage(imageData, d);
    
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            if (imageData[x][y] != mirroredData[x][y + 1] || imageData[x][y] != mirroredData[x + 1][y] ||
                imageData[x][y] != mirroredData[x + 1][y + 2] || imageData[x][y] != mirroredData[x + 2][y + 1]) {
                F[x][y] = 0;
                dist[x + 1][y + 1] = 0.0; 
            }
            else {
                dist[x + 1][y + 1] = eps; 
            }
        }
    }

    for (int t = 0; t < maxIter; t++) {
        int count = 0;
        for (int x = 0; x < height; x++) {
            for (int y = 0; y < width; y++) {
                if (F[x][y] == 1) {
                    double a = pow(std::min(dist[x][y + 1] - dist[x + 1][y + 1], 0.0), 2);
                    double b = pow(std::min(dist[x + 2][y + 1] - dist[x + 1][y + 1], 0.0), 2);
                    double c = pow(std::min(dist[x + 1][y] - dist[x + 1][y + 1], 0.0), 2);
                    double d = pow(std::min(dist[x + 1][y + 2] - dist[x + 1][y + 1], 0.0), 2);
                    double oldDist = dist[x + 1][y + 1];
                    dist[x+1][y+1] = oldDist + tau - (tau / h) * sqrt(std::max(a, b) + std::max(c, d));
                    if (std::abs(dist[x + 1][y + 1] - oldDist) < eps) {
                        F[x][y] = 0;
                    }
                    count++;
                }
            }
        }

        for (int x = 1; x <= height; x++) {
            dist[x][0] = dist[x][1];
            dist[x][width + 1] = dist[x][width];
        }
        for (int y = 1; y <= width; y++) {
            dist[0][y] = dist[1][y];
            dist[height + 1][y] = dist[height][y];
        }
        dist[0][0] = dist[1][1];
        dist[0][width + 1] = dist[1][width];
        dist[height + 1][0] = dist[height][1];
        dist[height + 1][width + 1] = dist[height][width];
        
        if (count == 0) {
            qDebug() << "casovy krok:" << t + 1;
            break;
        }
    }

    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            outputData[x][y] = dist[x + 1][y + 1];
        }
    }

    outputData = normalizeDoubleMatrix(outputData);

    return doubleToImage(outputData, width, height);
}