```markdown
# 🦵 OSTEO VISION - AI-Powered Knee Osteoarthritis Detection

A deep learning-powered Streamlit application for automated detection and severity grading of knee osteoarthritis from X-ray images using the Kellgren-Lawrence (KL) grading system.

## 🌟 Features

- **Automated KL Grading**: Classifies knee X-rays into 5 severity levels (KL Grade 0-4)
- **Deep Learning Model**: Fine-tuned ResNet152V2 architecture
- **Explainable AI**: Grad-CAM visualization showing which regions influenced the prediction
- **Interactive Web Interface**: Easy-to-use Streamlit frontend
- **Confidence Scores**: Probability distribution across all KL grades
- **Auto-Download**: Model automatically downloads from Google Drive on first run

## 📊 Kellgren-Lawrence Grading System

| Grade | Severity | Description |
|-------|----------|-------------|
| **KL-0** | Normal | No radiographic features of OA |
| **KL-1** | Doubtful | Minute osteophyte, doubtful significance |
| **KL-2** | Minimal | Definite osteophyte, unimpaired joint space |
| **KL-3** | Moderate | Moderate diminution of joint space |
| **KL-4** | Severe | Joint space greatly impaired with sclerosis |

## 🏗️ Project Structure

```
Osteo-Vision/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── models/
│   └── osteo_vision_model.h5  # Downloaded model (auto-downloaded)
├── .gitignore
└── README.md              # Documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- GPU optional (for faster inference)

### Setup Instructions

1. **Clone the repository**
   ```
   git clone https://github.com/vivekcm143/Osteo-Vision.git
   cd Osteo-Vision
   ```

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```
   streamlit run app.py
   ```

The trained model will automatically download from Google Drive on first run (~200-300 MB).

## 💻 Usage

1. **Launch the app**
   ```
   streamlit run app.py
   ```

2. **Upload X-ray image**
   - Click "Browse files" in the sidebar
   - Select a knee X-ray image (PNG, JPG, or JPEG)

3. **View results**
   - **Prediction**: Shows the predicted KL grade with confidence
   - **Original Image**: Your uploaded X-ray
   - **Grad-CAM Heatmap**: Visual explanation of the prediction
   - **Confidence Chart**: Probability distribution across all grades

## 🧠 Model Architecture

- **Base Model**: ResNet152V2 (pre-trained on ImageNet)
- **Fine-tuning**: Transfer learning on knee osteoarthritis dataset
- **Input Size**: 224×224 RGB images
- **Output**: 5-class classification (KL Grades 0-4)
- **Framework**: TensorFlow/Keras

### Model Details
```
Architecture: ResNet152V2
Input Shape: (224, 224, 3)
Total Parameters: ~60M
Trainable Parameters: Fine-tuned layers
Classes: 5 (KL-GRADE 0 to KL-GRADE 4)
```

## 🔬 Grad-CAM Explainability

The application uses **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualize:
- Which regions of the X-ray influenced the prediction
- Areas of joint space narrowing, osteophytes, or sclerosis
- Helps validate model decisions and build trust

## 📁 Model Download

The model is hosted on Google Drive and will automatically download on first run.

**Manual download option:**
- [Download Model (osteo_vision_model.h5)](https://drive.google.com/file/d/1hQ-H_GhruF1_Nkalle4DpHw4N6CKxCvD/view?usp=sharing)
- Place in `models/` directory

## 🎯 Performance Metrics

*(Add your model performance after evaluation)*

Example:
```
Overall Accuracy: 87%
Precision: 85%
Recall: 86%
F1-Score: 85%

Per-class Performance:
KL-0: 92% accuracy
KL-1: 84% accuracy
KL-2: 86% accuracy
KL-3: 88% accuracy
KL-4: 90% accuracy
```

## 🔧 Troubleshooting

**Issue**: Model download fails
- **Solution**: Download manually from the link above and place in `models/` folder

**Issue**: TensorFlow not using GPU
- **Solution**: Install `tensorflow-gpu` and ensure CUDA is properly configured

**Issue**: Out of memory error
- **Solution**: Close other applications or use a machine with more RAM

**Issue**: Image preprocessing error
- **Solution**: Ensure uploaded image is a valid knee X-ray in PNG/JPG format

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Deep Learning**: TensorFlow/Keras
- **Computer Vision**: OpenCV, Pillow
- **Visualization**: Matplotlib
- **Model Storage**: Google Drive with gdown

## 📈 Future Enhancements

- [ ] Support for batch processing of multiple X-rays
- [ ] Export detailed PDF reports
- [ ] Add temporal analysis (track progression over time)
- [ ] Multi-view support (AP and lateral views)
- [ ] Integration with PACS systems
- [ ] Mobile app deployment
- [ ] REST API for clinical integration

## ⚠️ Disclaimer

This application is intended for **research and educational purposes only**. It should **NOT** be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for medical advice.

## 📚 Dataset & Training

- **Dataset**: Custom knee osteoarthritis X-ray dataset
- **Augmentation**: Rotation, flipping, brightness adjustment
- **Validation Strategy**: K-fold cross-validation
- **Training Framework**: TensorFlow 2.x with mixed precision training

## 🎓 Academic Context

This project was developed as part of medical imaging research in AI/ML for healthcare applications.

**Author**: Vivek C M
- Education: BE in Artificial Intelligence and Machine Learning, VTU Belagavi
- GitHub: [@vivekcm143](https://github.com/vivekcm143)

## 📖 References

1. Kellgren, J. H., & Lawrence, J. S. (1957). Radiological assessment of osteo-arthrosis. *Annals of the rheumatic diseases*.
2. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *ICCV*.
3. He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

## 📄 License

MIT License

## 📧 Contact

For questions, collaboration, or feedback:
- Open an issue on GitHub
- Contact via GitHub profile: [@vivekcm143](https://github.com/vivekcm143)

---

⭐ **If you find this project helpful, please star the repository!**

## 🖼️ Screenshots

*(Add screenshots of your application here)*

### Main Interface
![Main Interface](screenshots/main_interface.png)

### Prediction Results
![Prediction](screenshots/prediction_result.png)

### Grad-CAM Visualization
![Grad-CAM](screenshots/gradcam_heatmap.png)
```

***