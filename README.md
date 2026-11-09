# Wine Quality Prediction System

## Project overview
This classroom prototype helps a small wine-production business preliminarily classify red-wine samples using physicochemical measurements. It predicts **GOOD QUALITY** or **NEEDS IMPROVEMENT**.

## Prediction rule
- Original quality score >= 6: `GOOD QUALITY` (class 1)
- Original quality score < 6: `NEEDS IMPROVEMENT` (class 0)

The original `quality` column is used only to create the target and is excluded from model input features to prevent target leakage.

## Dataset
UCI Machine Learning Repository — Wine Quality (red wine CSV):
https://archive.ics.uci.edu/dataset/186/wine+quality

## Technologies
Python, pandas, scikit-learn, Random Forest, joblib, Streamlit, Git/GitHub.

## Setup
Use Python 3.10 or later. In a terminal in this folder:

```bash
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The training script downloads the dataset, saves `wine_quality.csv`, trains and evaluates the model, and saves `wine_model.pkl`. Streamlit opens at the local address shown in the terminal (normally http://localhost:8501).

## App use
1. Run the training script before starting the app.
2. Download the sample input CSV from the app, or prepare a CSV with all 11 predictor columns.
3. Upload the CSV. The app checks required columns and displays predictions.
4. Download the prediction results CSV.

## SAD mini-analysis
- **Problem:** A producer needs a consistent preliminary way to screen wine samples using recorded physicochemical measurements.
- **Users:** Quality-control staff, production supervisor, or data analyst.
- **Inputs:** Fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol.
- **Process:** Validate required columns → preprocess measurements → apply trained model → classify the quality category.
- **Outputs:** GOOD QUALITY / NEEDS IMPROVEMENT label and model evaluation summary.
- **Limitation:** The model is trained on a particular red-wine dataset and may not generalize to other wine types, producers, regions, or testing conditions.

## Testing checklist
- [ ] `python train_model.py` downloads dataset and creates model file.
- [ ] Model evaluation prints accuracy, precision, recall, F1-score, and confusion matrix.
- [ ] `streamlit run app.py` opens the dashboard.
- [ ] Sample CSV upload produces predictions.
- [ ] CSV missing a required column shows a helpful error.
- [ ] Prediction results can be downloaded.
- [ ] GitHub repository contains source files and README.

## Reflection answers
1. **What is the prediction target, and why is quality excluded from inputs?**  
   The target is the binary quality category: quality score 6 or higher is GOOD QUALITY, and below 6 is NEEDS IMPROVEMENT. The original quality column is excluded because it is used to create the target. Including it as an input would reveal the answer to the model and cause target leakage.

2. **What does recall for GOOD QUALITY tell you? What could low recall mean for the business?**  
   Recall for GOOD QUALITY tells how many actual good-quality samples the model correctly identifies. Low recall means some good samples are classified as needing improvement, which may cause extra reviews or missed opportunities.

3. **Why might the model perform differently on another producer's wine?**  
   Different producers, regions, grapes, equipment, and testing conditions can produce different measurement patterns. The training dataset may not represent those samples.

4. **Why is this preliminary screening and not a final quality decision?**  
   The model is a classroom prototype trained on a specific public dataset. It does not replace laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control decisions.

5. **What additional data or testing could improve reliability?**  
   More samples from different producers, regions, vintages, and conditions; consistent laboratory measurements; external testing on new samples; and repeated evaluation with suitable metrics could help assess and improve reliability.

## GitHub publishing
Create a repository named `wine-quality-prediction`, then run:

```bash
git init
git add train_model.py app.py requirements.txt README.md
git commit -m "Build wine quality prediction prototype"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/wine-quality-prediction.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. Do not put passwords or access tokens in source code.

## Responsible-use note
This is a classroom prototype based on a public dataset. Predictions are preliminary screening results, not a substitute for laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control decisions.
