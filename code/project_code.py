# Below required libraries are imported

from nltk.corpus import stopwords  # Importing the stopwords
from nltk.tokenize import word_tokenize  # Library required to tokenize the words in a text
from nltk.stem import PorterStemmer  # Library to perform Stemming
from nltk.stem import WordNetLemmatizer  # Library to perform Lemmatization
from sklearn.feature_extraction.text import TfidfVectorizer  # Library to extract features from text
from sklearn.model_selection import KFold  # Library to perform splitting of data
from sklearn.decomposition import PCA  # Library to perform PCA to reduce the dimensions of data
from sklearn import svm  # Library to import SVM classifier
from sklearn.tree import DecisionTreeClassifier  # Library to import Decision tree classifier
from sklearn.metrics import classification_report, \
    f1_score  # Library to import evaluation parameters classification report and f1 score
from sklearn.metrics import confusion_matrix  # Library to import evaluation parameter confusion matrix
from sklearn.metrics import accuracy_score  # Library to import evaluation parameter accuracy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Library to plot the evaluation parameters graphically

print("Program execution started...")
print('-' * 50)

# Opening the text files from dataset
fn = open("non_clickbait_data.txt", "r")
fc = open("clickbait_data.txt", "r")


# Reading the articles from dataset line by line
f1 = fn.readlines()
f2 = fc.readlines()
rows = []
rows1 = []
stopwords = set(stopwords.words('english'))
acc = []


# Function to remove stopwords from the sentence and return the sentence without stopwords
def stopword(paragraph):
    words_tokens = word_tokenize(paragraph)  # Tokenizing the sentence
    sentence = []
    for w in words_tokens:
        if w not in stopwords:
            sentence.append(w)  # Removing the stopwords from sentence
    return sentence

#Function to perform stemming
def stemlemm(sentence):
    words_tokens = word_tokenize(sentence)
    filtered_sentence_list = [w for w in words_tokens if w not in stopwords]

    StemSentence = []
    LemmSentence = []
    Stemmed_Sentence = []
    for word in filtered_sentence_list:
        StemSentence.append(PorterStemmer().stem(word))  # Performing stemming

    for wor in StemSentence:
        LemmSentence.append(WordNetLemmatizer().lemmatize(wor))  # Performing Lemmatization

    Stemmed_Sentence = " ".join(StemSentence)
    Lemmatized_Sentence = " ".join(LemmSentence)

    return Lemmatized_Sentence  # Return Lemmatized sentence

#This function creates dataframe containing dataset and processed dataset
def CreateDataframe():

    #Below loop creates rows in dataframe with data from non-clickbait dataset

    for x in f1:
        if x != "\n":
            rows.append({"Headline": x, "Label": "non_clickbait"})
            LemmatizedSentence = stemlemm(x)

            Sentencewostopwords = stopword(LemmatizedSentence)
            Sentencewostopwords = ' '.join(Sentencewostopwords)

            rows1.append({"Headline": Sentencewostopwords,
                          "Label": "non_clickbait"})  # Creating rows with non-clickbait processed data
    # Below loop creates rows in dataframe with data from clickbait dataset
    for y in f2:
        if y != "\n":
            rows.append({"Headline": y, "Label": "clickbait"})
            LemmatizedSentence = stemlemm(y)
            Sentencewostopwords = stopword(LemmatizedSentence)
            Sentencewostopwords = ' '.join(Sentencewostopwords)
            rows1.append(
                {"Headline": Sentencewostopwords, "Label": "clickbait"})  # Creating rows with clickbait processed data


    # print(z)
    # print(rows)
    DataFrameColumns = ["Headline", "Label"]  # Labelling the dataframe columns
    DataFrameCustom1 = pd.DataFrame(rows, columns=DataFrameColumns)  # Creating dataframe with original text data
    DataFrameCustom2 = pd.DataFrame(rows1, columns=DataFrameColumns)  # Creating dataframe with processed data
    DataFrameCustom1 = DataFrameCustom1.sample(frac=1)  # Shuffling data in the original dataframe
    DataFrameCustom2 = DataFrameCustom2.sample(frac=1)  # Shuffling data in the processed dataframe
    DataFrameCustom1.to_csv('original_data.csv')  # Creating CSV file with original dataframe
    print("Original dataframe with data from text files created.")  # Creating dataframe with processed dataframe

    DataFrameCustom2.to_csv('processed_data.csv')
    print("Processed dataframe created after removing stopwords and performing lemmatization..")

    return DataFrameCustom2


# Function to extract features from the text data
def FeatureExtraction(DF):
    print("Extracting features from the text data...")
    textdata = DF["Headline"]
    lbl = DF["Label"]
    LabelArray = pd.Series(lbl).values  # Creating a series of labels
    vectorizer = TfidfVectorizer()  # Initializing TF-IDF vectorizer
    FeatureData = vectorizer.fit_transform(textdata).toarray()  # Extarcting features from the text
    FeatureDataFrameData = np.column_stack(
        (FeatureData, LabelArray))  # Creating datframe with features and their respective labels
    dataFrameColumns = vectorizer.get_feature_names()
    dataFrameColumns.insert(len(dataFrameColumns), 'Label')
    FeatureDataFrame = pd.DataFrame(data=FeatureDataFrameData,
                                    columns=dataFrameColumns)  # This dataframe gives the data with extracted features and labels
    return FeatureDataFrame

#Below function performs classification of labels
def Classification(DF):
    X = DF.iloc[:, :-1].values
    y = DF['Label']
    print("Splitting the data into training and test data using K-Fold....")

    kf = KFold(n_splits=2)  # Initializing K-Fold split
    for train, test in kf.split(X, y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[
            test]  # Splitting the data into test data and train data


    # PCA
    pca = PCA(n_components=70)
    X_train = pca.fit_transform(X_train)  # Performing Principle Component analysis
    X_test = pca.transform(X_test)
    print("Classification using SVM started.....")

    # SVM Classifier
    SVC_model = svm.SVC(kernel='linear', C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                        decision_function_shape='ovr', degree=3, gamma='auto',
                        max_iter=-1, probability=False, random_state=None,
                        shrinking=True, tol=0.001, verbose=False)  # SVM classifier with all the parameters tuned
    SVC_model.fit(X_train, y_train)  # Applying SVM classifier to training data
    SVC_prediction = SVC_model.predict(X_test)  # Predicting the label from test set
    print("Metrics of SVM are")
    qualityofclassifier(SVC_prediction, y_test)  # Calling the function to evaluate the performance of classifier

    # DecisionTreeClassifier
    print("Classification using Decision Tree started......")
    DT_gini = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=5, max_features=10,
                                     max_leaf_nodes=None, min_samples_leaf=1,
                                     min_samples_split=2,
                                     min_weight_fraction_leaf=0.0, random_state=None,
                                     splitter='best')  # Decision Tree classifier with all the parameters tuned
    DT_gini.fit(X_train, y_train)  # Applying SVM clasifier to training data
    DT_pred = DT_gini.predict(X_test)  # Predicting the label from test set
    print("Metrics of DT are")
    qualityofclassifier(DT_pred, y_test)  # Calling the function to evaluate the performance of classifier

#Below function is used to evaluate quality of the classifier
def qualityofclassifier(prediction, testlabel):
    print("Confusion Matrix :" + str(confusion_matrix(prediction, testlabel)))  # Evaluating confusion matrix
    print("Classification Report: " + str(
        classification_report(prediction, testlabel)))  # Evaluating classification report
    acc.append(accuracy_score(prediction, testlabel))  # Appending accuracy values to plot graphically
    print("Accuracy is: " + str(accuracy_score(prediction, testlabel)))  # Calculating accuracy
    print("F1 score is: " + str(f1_score(prediction, testlabel, average="weighted")))  # Calculating F1-score

#Below function is used to plot accuracy as bar graphs
def plot_bar_x():
    # this is for plotting purpose
    label = ['SVM Classifier', 'Decision Tree Classifier']
    index = np.arange(len(label))
    global rects1
    rects1 = plt.bar(index, acc, width=0.45, align='center')
    plt.xlabel('Classifier', fontsize=15)  # Labeling X-Axis
    plt.ylabel('Accuracy', fontsize=15)  # Labeling Y-Axis
    plt.xticks(index, label, fontsize=10, rotation=30)  # Plotting bar graph as per accuracy values
    plt.title('Accuracy comparison')
    autolabel(rects1)  # Labeling the bar graphs
    plt.show()

#Below function is used to label the bar graphs
def autolabel(rects, xpos='center'):
    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() * offset[xpos], 1.01 * height,
                 '{}'.format(height), ha=ha[xpos], va='bottom')  # Appending text to bars


processedDF = CreateDataframe()  # Calling the function which generates processed dataframe from the text files
featDF = FeatureExtraction(processedDF)  # Calling the function for text feature extraction
Classification(featDF)  # Calling the function to classify the labels
plot_bar_x()  # Calling the function to plot the accuracy values graphically(Graphs cannot be viewed in bluenose)
