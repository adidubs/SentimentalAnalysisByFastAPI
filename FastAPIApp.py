import joblib
import sentimental as a
# After training, save the models
joblib.dump(a.vectorizer, 'vectorizer.joblib')
joblib.dump(a.nb_classifier, 'nb_classifier.joblib')
joblib.dump(a.svm_classifier, 'svm_classifier.joblib')
joblib.dump(a.rf_classifier, 'rf_classifier.joblib')

vectorizer = joblib.load('vectorizer.joblib')
nb_classifier = joblib.load('nb_classifier.joblib')
svm_classifier = joblib.load('svm_classifier.joblib')
rf_classifier = joblib.load('rf_classifier.joblib')
