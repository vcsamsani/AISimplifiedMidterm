import joblib
import os
import json
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pickle
import boto3

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError("This model only supports application/json input")


def predict_fn(input_data, model):
    # Process the input data if necessary
    processed_data = process_input(input_data)
    # Make predictions using the model
    predictions = model.predict(processed_data)
    #print(predictions)
    return predictions

def process_input(input_data):
    # Process input data as needed before passing to the model for prediction
    NgramFeaturesList_pred = np.array(input_data['NgramFeaturesList_pred'])
    importsCorpus_pred = input_data['importsCorpus_pred']
    sectionNames_pred = input_data['sectionNames_pred']
    numSections_pred = int(input_data['numSections_pred'])
    

    # Load featurizers
    imports_featurizer = joblib.load(os.path.join("opt/ml/model", "imports_featurizer.pkl"))
    section_names_featurizer = joblib.load(os.path.join("opt/ml/model", "section_names_featurizer.pkl"))
    #print(NgramFeaturesList_pred, importsCorpus_pred, sectionNames_pred, numSections_pred)
    #print(imports_featurizer, section_names_featurizer)
    # Transform text features
    importsCorpus_pred_transformed = imports_featurizer.transform([importsCorpus_pred])
    sectionNames_pred_transformed = section_names_featurizer.transform([sectionNames_pred])

    # Concatenate features into a single sparse matrix
    processed_data = hstack([csr_matrix(NgramFeaturesList_pred),
                             importsCorpus_pred_transformed,
                             sectionNames_pred_transformed,
                             csr_matrix([numSections_pred]).transpose()])
    #print(processed_data)
    return processed_data


def output_fn(prediction, content_type):
    res = int(prediction[0])
    respJSON = {'Output': res}
    return respJSON
