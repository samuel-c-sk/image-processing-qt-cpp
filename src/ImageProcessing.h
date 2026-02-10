#pragma once
#include <QImage>
#include <vector>

class ImageProcessing {

private:
	static std::vector<std::vector<double>> imageToDouble(const QImage& image);
	static QImage doubleToImage(const std::vector<std::vector<double>>& imageData, int width, int height);
public:
	static QImage invertColors(const QImage& image);
	static QImage FSHS(const QImage& image);
	static QImage EH(const QImage& image);
	static QImage linearConvolution(const QImage& image);
	static QImage explicit_scheme(const QImage& image, int T, double tau);
	static QImage implicit_scheme(const QImage& image, int T, double tau);
	static QImage EdgesDetection(const QImage& image, int K);
	static QImage PeroneMalik(const QImage& image, int T, double tau, double sigma, int K);
	static QImage MCF(const QImage& image, int T, double tau, double epsilon);
	static QImage GMCF(const QImage& image, int T, double tau, double sigma, int K, double epsilon);
	static QImage Rouy_Tourin(const QImage& image, double tau, int K);

	static QImage mirrorImage(const QImage& image, int d);
	static std::vector<std::vector<double>> mirrorImage(const std::vector<std::vector<double>>& image, int d);
	static std::vector<std::vector<double>> normalizeDoubleMatrix(const std::vector<std::vector<double>>& image);
};

//static - nie je nutne volat funkcie cez instanciu triedy ale cez samotnu triedu, so static by bolo nutne vytvorit najprv objekt triedy Imageprocessing