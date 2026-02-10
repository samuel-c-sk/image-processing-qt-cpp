#include "ImageViewer.h"

ImageViewer::ImageViewer(QWidget* parent)
	: QMainWindow(parent), ui(new Ui::ImageViewerClass)
{
	ui->setupUi(this);
	vW = new ViewerWidget(QSize(500, 500));
	ui->scrollArea->setWidget(vW);

	ui->scrollArea->setBackgroundRole(QPalette::Dark);
	ui->scrollArea->setWidgetResizable(true);
	ui->scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
	ui->scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

	vW->setObjectName("ViewerWidget");
}

//ImageViewer Events
void ImageViewer::closeEvent(QCloseEvent* event)
{
	if (QMessageBox::Yes == QMessageBox::question(this, "Close Confirmation", "Are you sure you want to exit?", QMessageBox::Yes | QMessageBox::No))
	{
		event->accept();
	}
	else {
		event->ignore();
	}
}

//Image functions
bool ImageViewer::openImage(QString filename)
{
	QImage loadedImg(filename);
	if (!loadedImg.isNull()) {
		originalImage = loadedImg;
		return vW->setImage(loadedImg);
	}
	return false;
}
bool ImageViewer::saveImage(QString filename)
{
	QFileInfo fi(filename);
	QString extension = fi.completeSuffix();

	QImage* img = vW->getImage();
	return img->save(filename, extension.toStdString().c_str());
}

//Slots
void ImageViewer::on_actionOpen_triggered()
{
	QString folder = settings.value("folder_img_load_path", "").toString();

	QString fileFilter = "Image data (*.bmp *.gif *.jpg *.jpeg *.png *.pbm *.pgm *.ppm .*xbm .* xpm);;All files (*)";
	QString fileName = QFileDialog::getOpenFileName(this, "Load image", folder, fileFilter);
	if (fileName.isEmpty()) { return; }

	QFileInfo fi(fileName);
	settings.setValue("folder_img_load_path", fi.absoluteDir().absolutePath());

	if (!openImage(fileName)) {
		msgBox.setText("Unable to open image.");
		msgBox.setIcon(QMessageBox::Warning);
		msgBox.exec();
	}
}
void ImageViewer::on_actionSave_as_triggered()
{
	QString folder = settings.value("folder_img_save_path", "").toString();

	QString fileFilter = "Image data (*.bmp *.gif *.jpg *.jpeg *.png *.pbm *.pgm *.ppm .*xbm .* xpm);;All files (*)";
	QString fileName = QFileDialog::getSaveFileName(this, "Save image", folder, fileFilter);
	if (!fileName.isEmpty()) {
		QFileInfo fi(fileName);
		settings.setValue("folder_img_save_path", fi.absoluteDir().absolutePath());

		if (!saveImage(fileName)) {
			msgBox.setText("Unable to save image.");
			msgBox.setIcon(QMessageBox::Warning);
		}
		else {
			msgBox.setText(QString("File %1 saved.").arg(fileName));
			msgBox.setIcon(QMessageBox::Information);
		}
		msgBox.exec();
	}
}
void ImageViewer::on_actionExit_triggered()
{
	this->close();
}

//Invrtovanie farieb
void ImageViewer::on_actionInvert_triggered()
{
	invertColors();
}
bool ImageViewer::invertColors()
{
	if (vW->isEmpty()) {
		return false;
	}

	QImage* img = vW->getImage();
	QImage processedImg = ImageProcessing::invertColors(*img);
	vW->setImage(processedImg);

	return true;
}

//pushbuttons
void ImageViewer::on_pushButtonFSHS_clicked()
{
	if (vW->isEmpty()) {
		return;
	}

	QImage* img = vW->getImage();
	QImage processedImg = ImageProcessing::FSHS(*img);
	vW->setImage(processedImg);
}

void ImageViewer::on_pushButtonEH_clicked()
{
	if (vW->isEmpty()) {
		return;
	}

	QImage* img = vW->getImage();
	QImage processedImg = ImageProcessing::EH(*img);
	vW->setImage(processedImg);
}

void ImageViewer::on_pushButtonLinCon_clicked()
{
	if (vW->isEmpty()) {
		return;
	}
	QImage* img = vW->getImage();
	QImage processedImg = ImageProcessing::linearConvolution(*img);
	vW->setImage(processedImg);
}

void ImageViewer::on_pushButtonRVT_clicked()
{
	bool ok;
	double tau = ui->lineEditTau->text().toDouble(&ok);
	if (vW->isEmpty()) {
		return;
	}
	QImage* img = vW->getImage();

	if (ui->spinBoxT->value() > 0) 
	{
		if (ok && tau > 0.0 && tau < 1.0) 
		{
			if (tau <= 0.25)
			{
				QImage processedImg = ImageProcessing::explicit_scheme(*img, ui->spinBoxT->value(), tau);
				vW->setImage(processedImg);
			}
			else 
			{
				QImage processedImg = ImageProcessing::implicit_scheme(*img, ui->spinBoxT->value(), tau);
				vW->setImage(processedImg);
			}
		}
		else {
			qDebug() << "Nespravna hodnota Tau";
		}
	}
	else {
		qDebug() << "nespravna hodnota T";
	}
}

void ImageViewer::on_pushButtonEdges_clicked()
{
	if (vW->isEmpty()) {
		return;
	}

	bool ok;
	int K = ui->lineEditK->text().toInt(&ok);

	QImage* img = vW->getImage();
	if (ok) {
		QImage processedImg = ImageProcessing::EdgesDetection(*img, K);
		vW->setImage(processedImg);
	}
}

void ImageViewer::on_pushButtonPM_clicked()
{
	if (vW->isEmpty()) {
		return;
	}

	bool ok, ok2, ok3;

	double tau = ui->lineEditTau->text().toDouble(&ok);
	double sigma = ui->lineEditSigma->text().toDouble(&ok2);
	int K = ui->lineEditK->text().toInt(&ok3);


	QImage* img = vW->getImage();

	if (ui->spinBoxT->value() > 0)
	{
		if (ok && tau > 0.0 && tau <= 1.0 && ok2 && sigma > 0.0 && sigma <= 1.0 && ok3)
		{
			QImage processedImg = ImageProcessing::PeroneMalik(*img, ui->spinBoxT->value(), tau, sigma, K);
			vW->setImage(processedImg);
		}
		else {
			qDebug() << "Nespravna hodnota Tau/Sigma";
		}
	}
	else {
		qDebug() << "nespravna hodnota T";
	}
}

void ImageViewer::on_pushButtonMCF_clicked()
{
	if (vW->isEmpty()) {
		return;
	}

	bool ok, ok4;

	double tau = ui->lineEditTau->text().toDouble(&ok);
	double epsilon = ui->lineEditEpsilon->text().toDouble(&ok4);


	QImage* img = vW->getImage();

	if (ui->spinBoxT->value() > 0)
	{
		if (ok && tau > 0.0 && tau <= 5.0 && ok4)
		{
			QImage processedImg = ImageProcessing::MCF(*img, ui->spinBoxT->value(), tau, epsilon);
			vW->setImage(processedImg);
		}
		else {
			qDebug() << "Nespravna hodnota Tau/epsilon";
		}
	}
	else {
		qDebug() << "nespravna hodnota T";
	}
}

void ImageViewer::on_pushButtonGMCF_clicked()
{
	if (vW->isEmpty()) {
		return;
	}

	bool ok, ok2, ok3, ok4;

	double tau = ui->lineEditTau->text().toDouble(&ok);
	double sigma = ui->lineEditSigma->text().toDouble(&ok2);
	int K = ui->lineEditK->text().toInt(&ok3);
	double epsilon = ui->lineEditEpsilon->text().toDouble(&ok4);



	QImage* img = vW->getImage();

	if (ui->spinBoxT->value() > 0)
	{
		if (ok && tau > 0.0 && tau <= 1.0 && ok2 && sigma > 0.0 && sigma <= 1.0 && ok3 && ok4)
		{
			QImage processedImg = ImageProcessing::GMCF(*img, ui->spinBoxT->value(), tau, sigma, K, epsilon);
			vW->setImage(processedImg);
		}
		else {
			qDebug() << "Nespravna hodnota Tau/Sigma/epsilon";
		}
	}
	else {
		qDebug() << "nespravna hodnota T";
	}
}

void ImageViewer::on_pushButtonRT_clicked()
{
	if (vW->isEmpty()) {
		return;
	}

	bool ok, ok3;

	double tau = ui->lineEditTau->text().toDouble(&ok);
	int K = ui->lineEditK->text().toInt(&ok3);

	QImage* img = vW->getImage();

	if (ok && tau > 0.0 && tau <= 0.5 && ok3)
		{
			QImage processedImg = ImageProcessing::Rouy_Tourin(*img, tau, K);
			vW->setImage(processedImg);
		}
	else {
		qDebug() << "Nespravna hodnota Tau";
	}
}

void ImageViewer::on_pushButtonReset_clicked()
{
	if (!originalImage.isNull()) {
		vW->setImage(originalImage);
	}
}
