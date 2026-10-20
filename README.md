# 🛡️ Spam Detector AI

A machine learning-powered spam message classifier built with **Flask**, **Scikit-learn**, and **NLTK**. It uses TF-IDF vectorization with a trained Logistic Regression model to detect spam in real time via a clean web interface.

## 🚀 Features

- Detects spam vs. ham (non-spam) messages with high accuracy
- Trained on multiple SMS datasets for better generalization
- Fast inference via pre-trained model pipeline
- Simple and intuitive web UI powered by Flask

## 🛠️ Tech Stack

| Layer       | Technology                     |
|-------------|-------------------------------|
| ML Model    | Scikit-learn (Logistic Regression) |
| Text Processing | NLTK, TF-IDF Vectorizer    |
| Web Server  | Flask                         |
| Data        | SMS Spam Collection Datasets  |

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/spam-detector.git
cd spam-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train_model.py
```
> This will generate the model files inside the `model/` directory.

### 4. Run the app
```bash
python app/main.py
```

Then open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

## 📁 Project Structure

```
spam-detector/
├── app/
│   ├── main.py            # Flask app entry point
│   ├── model_loader.py    # Loads the trained model
│   ├── preprocess.py      # Text preprocessing utilities
│   └── templates/
│       └── index.html     # Web UI
├── data/                  # Training datasets (excluded from git)
├── model/                 # Saved model files (excluded from git)
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
└── README.md
```

## 📝 Notes

- Model `.pkl` files and dataset `.csv` files are excluded from this repository due to their size.
- Run `train_model.py` to regenerate the model locally after cloning.

## 📄 License

MIT License
