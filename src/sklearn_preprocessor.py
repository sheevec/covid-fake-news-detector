from __future__ import print_function

import time
import sys
from io import StringIO
import os
import shutil

import argparse
import csv
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, StandardScaler, OneHotEncoder

from subprocess import call 
call("pip install nltk".split(" "))
call("pip install textstat".split(" "))
call("pip install lexicalrichness".split(" "))
call("pip install textblob".split(" "))

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords    
stop_words = set(stopwords.words('english'))

from nltk.tag import pos_tag
from nltk import word_tokenize
from collections import Counter

import textstat
from lexicalrichness import LexicalRichness

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Since we get a headerless CSV file we specify the column names here.
feature_columns_names = [
    'title', # Title of the article
    'text', # Article write up
    'source'] # source URL

label_column = 'label'

feature_columns_dtype = {
    'title': "object",
    'text': "object",
    'source': "object"}

label_column_dtype = {'label': "object"} # +1.5 gives the age in years

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    
    raw_data = [ pd.read_csv(
        file, 
        header=None, 
        names=feature_columns_names + [label_column]) for file in input_files ]
    concat_data = pd.concat(raw_data)

    concat_data.loc[concat_data['label'] == 'Fake', ['label']] = 'FAKE'
    concat_data.loc[concat_data['label'] == 'fake', ['label']] = 'FAKE'
    concat_data.loc[concat_data['source'] == 'facebook', ['source']] = 'Facebook'
    concat_data.head()
    concat_data.text.fillna(concat_data['title'], inplace=True)

    concat_data.loc[5]['label'] = 'FAKE'
    concat_data.loc[15]['label'] = 'TRUE'
    concat_data.loc[43]['label'] = 'FAKE'
    concat_data.loc[131]['label'] = 'TRUE'
    concat_data.loc[242]['label'] = 'FAKE'

    concat_data = concat_data.sample(frac=1).reset_index(drop=True)
    concat_data.title.fillna('missing', inplace=True)
    concat_data.source.fillna('missing', inplace=True)

    concat_data['title_num_uppercase'] = concat_data['title'].str.count(r'[A-Z]')
    concat_data['text_num_uppercase'] = concat_data['text'].str.count(r'[A-Z]')
    concat_data['text_len'] = concat_data['text'].str.len()
    concat_data['text_pct_uppercase'] = concat_data.text_num_uppercase.div(concat_data.text_len)

    concat_data['title_num_stop_words'] = concat_data['title'].str.split().apply(lambda x: len(set(x) & stop_words))
    concat_data['text_num_stop_words'] = concat_data['text'].str.split().apply(lambda x: len(set(x) & stop_words))
    concat_data['text_word_count'] = concat_data['text'].apply(lambda x: len(str(x).split()))
    concat_data['text_pct_stop_words'] = concat_data['text_num_stop_words'] / concat_data['text_word_count']

    concat_data.drop(['text_num_uppercase', 'text_len', 'text_num_stop_words', 'text_word_count'], axis=1, inplace=True)

    concat_data['token'] = concat_data.apply(lambda row: nltk.word_tokenize(row['title']), axis=1)
    concat_data['pos_tags'] = concat_data.apply(lambda row: nltk.pos_tag(row['token']), axis=1)

    tag_count_concat_data = pd.DataFrame(concat_data['pos_tags'].map(lambda x: Counter(tag[1] for tag in x)).to_list())
    concat_data = pd.concat([concat_data, tag_count_concat_data], axis=1).fillna(0).drop(['pos_tags', 'token'], axis=1)
    
    concat_data['num_negation'] = concat_data['text'].str.lower().str.count("no|not|never|none|nothing|nobody|neither|nowhere|hardly|scarcely|barely|doesn’t|isn’t|wasn’t|shouldn’t|wouldn’t|couldn’t|won’t|can't|don't")
    
    concat_data['num_interrogatives_title'] = concat_data['title'].str.lower().str.count("what|who|when|where|which|why|how")
    concat_data['num_interrogatives_text'] = concat_data['text'].str.lower().str.count("what|who|when|where|which|why|how")

    reading_ease = []
    for doc in concat_data['text']:
        reading_ease.append(textstat.flesch_reading_ease(doc))

    smog = []
    for doc in concat_data['text']:
        smog.append(textstat.smog_index(doc))

    kincaid_grade = []
    for doc in concat_data['text']:
        kincaid_grade.append(textstat.flesch_kincaid_grade(doc))

    liau_index = []
    for doc in concat_data['text']:
        liau_index.append(textstat.coleman_liau_index(doc))

    readability_index = []
    for doc in concat_data['text']:
        readability_index.append(textstat.automated_readability_index(doc))

    readability_score = []
    for doc in concat_data['text']:
        readability_score.append(textstat.dale_chall_readability_score(doc))

    difficult_words = []
    for doc in concat_data['text']:
        difficult_words.append(textstat.difficult_words(doc))

    write_formula = []
    for doc in concat_data['text']:
        write_formula.append(textstat.linsear_write_formula(doc))

    gunning_fog = []
    for doc in concat_data['text']:
        gunning_fog.append(textstat.gunning_fog(doc))

    text_standard = []
    for doc in concat_data['text']:
        text_standard.append(textstat.text_standard(doc))

    concat_data['flesch_reading_ease'] = reading_ease
    concat_data['smog_index'] = smog
    concat_data['flesch_kincaid_grade'] = kincaid_grade
    concat_data['automated_readability_index'] = readability_index
    concat_data['dale_chall_readability_score'] = readability_score
    concat_data['difficult_words'] = difficult_words
    concat_data['linsear_write_formula'] = write_formula
    concat_data['gunning_fog'] = gunning_fog
    concat_data['text_standard'] = text_standard

    ttr = []
    for doc in concat_data['text']:
        lex = LexicalRichness(doc)
        ttr.append(lex.ttr)

    concat_data['ttr'] = ttr

    concat_data['num_powerWords_text'] = concat_data['text'].str.lower().str.count('improve|trust|immediately|discover|profit|learn|know|understand|powerful|best|win|more|bonus|exclusive|extra|you|free|health|guarantee|new|proven|safety|money|now|today|results|protect|help|easy|amazing|latest|extraordinary|how to|worst|ultimate|hot|first|big|anniversary|premiere|basic|complete|save|plus|create')
    concat_data['num_casualWords_text'] = concat_data['text'].str.lower().str.count('make|because|how|why|change|use|since|reason|therefore|result')
    concat_data['num_tentativeWords_text'] = concat_data['text'].str.lower().str.count('may|might|can|could|possibly|probably|it is likely|it is unlikely|it is possible|it is probable|tends to|appears to|suggests that|seems to')
    concat_data['num_emotionWords_text'] = concat_data['text'].str.lower().str.count('ordeal|outrageous|provoke|repulsive|scandal|severe|shameful|shocking|terrible|tragic|unreliable|unstable|wicked|aggravate|agony|appalled|atrocious|corruption|damage|disastrous|disgusted|dreaconcat_dataul|eliminate|harmful|harsh|inconsiderate|enraged|offensive|aggressive|frustrated|controlling|resentful|anger|sad|fear|malicious|infuriated|critical|violent|vindictive|furious|contrary|condemning|sarcastic|poisonous|jealous|retaliating|desperate|alienated|unjustified|violated')
    
    concat_data.drop(['title', 'text', 'source', 'label', 'text_standard'], axis=1, inplace=True)

    numeric_transformer = make_pipeline(StandardScaler())

    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, make_column_selector(dtype_exclude="category"))])
    
    preprocessor.fit(concat_data)

    joblib.dump(preprocessor, os.path.join(args.model_dir, "model.joblib"))

    print("saved model!")
    
    
def input_fn(input_data, content_type):
    """Parse input data payload
    
    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    if content_type == 'text/csv':
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data), 
                         header=None)
        
        if len(df.columns) == len(feature_columns_names) + 1:
            # This is a labelled example, includes the ring label
            df.columns = feature_columns_names + [label_column]
        elif len(df.columns) == len(feature_columns_names):
            # This is an unlabelled example.
            df.columns = feature_columns_names
            
        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))
        

def output_fn(prediction, accept):
    """Format prediction output
    
    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))


def predict_fn(input_data, model):
    """Preprocess input data
    
    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:
    
        rest of features either one hot encoded or standardized
    """
    features = model.transform(input_data)

    if label_column in input_data:
        # Return the label (as the first column) and the set of features.
        return np.insert(features, 0, input_data[label_column], axis=1)
    else:
        # Return only the set of features
        return features
    

def model_fn(model_dir):
    """Deserialize fitted model
    """
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor
