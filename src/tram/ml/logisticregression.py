import logging
import pathlib
import pickle
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from io import BytesIO
from os import path

import docx
import nltk
import pdfplumber
from bs4 import BeautifulSoup
from constance import config
from django.conf import settings
from django.db import transaction
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

# The word model is overloaded in this scope, so a prefix is necessary
from tram import models as db_models

logger = logging.getLogger(__name__)

class SKLearnModel(ABC):
    """
    TODO:
    1. Move text extraction and tokenization out of the SKLearnModel
    """

    def __init__(self):
        self.techniques_model = self.get_model()
        self.last_trained = None
        self.average_f1_score = None
        self.detailed_f1_score = None

        if not isinstance(self.techniques_model, Pipeline):
            raise TypeError(
                "get_model() must return an sklearn.pipeline.Pipeline instance"
            )

    @abstractmethod
    def get_model(self):
        """Returns an sklearn.Pipeline that has fit() and predict() methods"""

    def train(self):
        """
        Load and preprocess data. Train model pipeline
        """
        X, y = self.get_training_data()

        self.techniques_model.fit(X, y)  # Train classification model
        self.last_trained = datetime.now(timezone.utc)

    def test(self):
        """
        Return classification metrics based on train/test evaluation of the data
        Note: potential extension is to use cross-validation rather than a single train/test split
        """
        x, y = self.get_training_data()

        # Create training set and test set
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, shuffle=True, random_state=0, stratify=y
        )

        # Train model
        test_model = self.get_model()
        test_model.fit(x_train, y_train)

        # Generate predictions on test set
        y_predicted = test_model.predict(x_test)

        # TODO: Does this put labels and scores in the correct order?
        # Calculate an f1 score for each technique
        labels = sorted(list(set(y)))
        scores = f1_score(y_test, y_predicted, labels=list(set(y)), average=None)
        self.detailed_f1_score = sorted(
            zip(labels, scores), key=lambda t: t[1], reverse=True
        )

        # Average F1 score across techniques, weighted by the # of training examples per technique
        weighted_f1 = f1_score(y_test, y_predicted, average="weighted")
        self.average_f1_score = weighted_f1

    def _get_report_name(self, job):
        name = pathlib.Path(job.document.docfile.path).name
        return "Report for %s" % name


    def lemmatize(self, sentence):
        """
        Preprocess text by
        1) Lemmatizing - reducing words to their root, as a way to eliminate noise in the text
        2) Removing digits
        """
        lemma = nltk.stem.WordNetLemmatizer()

        # Lemmatize each word in sentence
        lemmatized_sentence = " ".join(
            [lemma.lemmatize(w) for w in sentence.rstrip().split()]
        )
        lemmatized_sentence = re.sub(
            r"\d+", "", lemmatized_sentence
        )  # Remove digits with regex

        return lemmatized_sentence

    def get_training_data(self):
        """
        returns a tuple of lists, X, y.
        X is a list of lemmatized sentences; y is a list of Attack Techniques
        """
        X = []
        y = []
        mappings = db_models.Mapping.get_accepted_mappings()
        for mapping in mappings:
            lemmatized_sentence = self.lemmatize(mapping.sentence.text)
            X.append(lemmatized_sentence)
            y.append(mapping.attack_object.attack_id)

        return X, y

    def get_attack_object_ids(self):
        objects = [
            obj.attack_id
            for obj in db_models.AttackObject.objects.all().order_by("attack_id")
        ]
        if len(objects) == 0:
            raise ValueError(
                "Zero techniques found. Maybe run `python manage.py attackdata load` ?"
            )
        return objects

    def get_mappings(self, sentence):
        """
        Use trained model to predict the technique for a given sentence.
        """
        mappings = []

        techniques = self.techniques_model.classes_
        probs = self.techniques_model.predict_proba([sentence])[
            0
        ]  # Probability is a range between 0-1

        # Create a list of tuples of (confidence, technique)
        confidences_and_techniques = zip(probs, techniques)
        for confidence_and_technique in confidences_and_techniques:
            confidence = confidence_and_technique[0] * 100
            attack_technique = confidence_and_technique[1]
            if confidence < config.ML_CONFIDENCE_THRESHOLD:
                # Ignore proposed mappings below the confidence threshold
                continue
            mapping = Mapping(confidence, attack_technique)
            mappings.append(mapping)

        return mappings

    def _sentence_tokenize(self, text):
        return nltk.sent_tokenize(text)

    def process_job(self, job):
        name = self._get_report_name(job)
        text = self._extract_text(job.document)
        sentences = self._sentence_tokenize(text)

        report_sentences = []
        order = 0
        for sentence in sentences:
            mappings = self.get_mappings(sentence)
            s = Sentence(text=sentence, order=order, mappings=mappings)
            order += 1
            report_sentences.append(s)

        report = Report(name, text, report_sentences)
        return report

    def save_to_file(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, filepath):
        with open(filepath, "rb") as f:
            model = pickle.load(f)  # nosec
            # accept risk until better design implemented
        assert cls == model.__class__
        return model


class LogisticRegressionModel(SKLearnModel):
    def get_model(self):
        """
        Modeling pipeline:
        1) Features = document-term matrix, with stop words removed from the term vocabulary.
        2) Classifier (clf) = multinomial logistic regression
        """
        return Pipeline(
            [
                (
                    "features",
                    CountVectorizer(lowercase=True, stop_words="english", min_df=3),
                ),
                ("clf", LogisticRegression()),
            ]
        )


class ModelManager(object):

    def __init__(self, model):

        model_filepath = self.get_model_filepath(LogisticRegressionModel)
        if path.exists(model_filepath):
            self.model = model_class.load_from_file(model_filepath)
            logger.info("%s loaded from %s", model_class.__name__, model_filepath)
        else:
            self.model = model_class()
            logger.info("%s loaded from __init__", model_class.__name__)

    def run_model(self, run_forever=False):
        while True:
            jobs = db_models.DocumentProcessingJob.objects.filter(
                status="queued"
            ).order_by("created_on")
            for job in jobs:
                filename = job.document.docfile.name
                logger.info("Processing Job #%d: %s", job.id, filename)
                try:
                    report = self.model.process_job(job)
                    with transaction.atomic():
                        self._save_report(report, job.document)
                        job.delete()
                    logger.info("Created report %s", report.name)
                except Exception as ex:
                    job.status = "error"
                    job.message = str(ex)
                    job.save()
                    logger.exception("Failed to create report for %s.", filename)

            if not run_forever:
                return
            time.sleep(1)

    def get_model_filepath(self, model_class):
        filepath = settings.ML_MODEL_DIR + "/" + model_class.__name__ + ".pkl"
        return filepath

    def train_model(self):
        self.model.train()
        self.model.test()
        filepath = self.get_model_filepath(self.model.__class__)
        self.model.save_to_file(filepath)
        logger.info("Trained model saved to %s" % filepath)


