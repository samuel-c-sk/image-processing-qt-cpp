#pragma once

#include <QtWidgets/QMainWindow>
#include <QtWidgets>
#include "ui_ImageViewer.h"
#include "ViewerWidget.h"
#include "ImageProcessing.h"


class ImageViewer : public QMainWindow
{
	Q_OBJECT

public:
	ImageViewer(QWidget* parent = Q_NULLPTR);

private:
	Ui::ImageViewerClass* ui;
	ViewerWidget* vW;

	QImage originalImage;

	QSettings settings;
	QMessageBox msgBox;

	//ImageViewer Events
	void closeEvent(QCloseEvent* event);

	//Image functions
	bool openImage(QString filename);
	bool saveImage(QString filename);
	bool invertColors();

private slots:
	void on_actionOpen_triggered();
	void on_actionSave_as_triggered();
	void on_actionExit_triggered();
	void on_actionInvert_triggered();
	
	void on_pushButtonReset_clicked();
	void on_pushButtonFSHS_clicked();
	void on_pushButtonEH_clicked();
	void on_pushButtonLinCon_clicked();
	void on_pushButtonRVT_clicked();
	void on_pushButtonEdges_clicked();
	void on_pushButtonPM_clicked();
	void on_pushButtonGMCF_clicked();
	void on_pushButtonMCF_clicked();
	void on_pushButtonRT_clicked();
};
