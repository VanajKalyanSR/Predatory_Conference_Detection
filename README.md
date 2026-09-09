# Predatory_Conference_Detection
1. Introduction
Predatory conferences are fraudulent academic events that exploit researchers by charging registration and publication fees while providing little or no legitimate peer review, editorial oversight, or genuine scientific value. These events mimic the appearance of legitimate conferences by using professional-sounding names, false indexing claims, and inflated impact factor figures to deceive academics, particularly early-career researchers.

The rise of predatory conferences poses a significant threat to academic integrity. Researchers who unknowingly submit to such events risk damaging their professional reputation, wasting financial resources, and contributing to the spread of unverified scientific claims. Manual identification of predatory conferences is time-consuming and inconsistent, creating a need for automated, data-driven detection systems.

This project develops an AI and Machine Learning-based system to automatically identify predatory conferences. The system is trained on scraped data from Beall's List (a well-known registry of predatory publishers) and the Directory of Open Access Journals (DOAJ), and is deployed as an interactive web application built using the Streamlit framework. The system achieves the target accuracy of above 95% through a stacking ensemble model combining XGBoost and Random Forest classifiers.


5.2  Technology Stack
Category	Library / Tool	Usage
Web Scraping	requests, BeautifulSoup4	HTTP requests and HTML parsing for data collection
Data Processing	pandas, numpy	Dataset manipulation, cleaning, and feature engineering
Machine Learning	scikit-learn, XGBoost	Model training, evaluation, and pipeline construction
Hyperparameter Tuning	Optuna	Automated XGBoost hyperparameter optimisation (50 trials)
Class Balancing	imbalanced-learn (SMOTE)	Synthetic oversampling during training
Explainability	SHAP	Feature importance visualisation for model interpretability
Model Persistence	joblib	Serialisation and loading of trained model artefacts
Web Interface	Streamlit	Interactive front-end for real-time conference analysis
Visualisation	matplotlib	Model comparison charts in Streamlit dashboard
Progress Tracking	tqdm	Progress bars during long scraping operations
