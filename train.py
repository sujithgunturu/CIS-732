import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tflearn
import tensorflow
import movie
import os
lemmatizer = WordNetLemmatizer()
def create_X_Y(classes, documents, words) :
    training = []
    #initialize to an array of zeros
    #try with np.zeros(len(classes))
    output_empty = [0] * len(classes)
    for doc in documents:
        bag = []
        question_words = doc[0]
        question_words = [lemmatizer.lemmatize(word.lower()) for word in question_words]
        for word in words:
            bag.append(1) if word in question_words else bag.append(0)   
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])
    import random
    random.shuffle(training)
    training = np.array(training)
    X = list(training[:,0])
    Y = list(training[:,1])
    return X, Y

def save_files(toDump, filename):
    import pickle
    pickle.dump(toDump,open( filename + '.pkl','wb'))

def compile_fit(X, Y):
    #https://keras.io/guides/sequential_model/
    from keras.models import Sequential as seq
    from keras.layers import Dense, Activation, Dropout
    from keras.optimizers import SGD
    model = seq()
    model.add(Dense(128, input_shape=(len(X[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(Y[0]), activation='softmax'))
    #compile the model with loss, nd sochastic gradient descent
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), metrics=['accuracy'])
    #convert the lists into nparrays
    x = np.array(X)
    y = np.array(Y)
    training_board = model.fit(x, y,validation_split=0.25, epochs=200,  batch_size=2, verbose=1)
    model.save('model.h5', training_board)
    return training_board, model

def plot_accuracyVSepochs(history):
    import matplotlib.pyplot as plt
    # plt.figure(1)
    # plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])
    plt.show()
    # plt.subplot(212)
    import matplotlib.pyplot as plt
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])
    plt.show()

def make_unique_and_sort(arratoList):
    arratoList = set(arratoList)
    arratoList = list(arratoList)
    arratoList = sorted(arratoList)
    return arratoList
def lemetize_maker(values, letterstoIngonre):
    values = [lemmatizer.lemmatize(v.lower()) for v in values if v not in letterstoIngonre]
    return values

def pre_process(filetoread, letterstoIngonre, nameOfPickle1, nameOfPickle2):
    #initializing emply arrays
    words=[]
    classes = []
    documents = []
    # getting the class names
    for category in filetoread['categories']:
        # getting the questions
        for question in category['questions']:
            #tokenize the word
            word = nltk.word_tokenize(question)
            words.extend(word)
            # input, label tuple
            documents.append((word, category['category']))
            if category['category'] not in classes:
                classes.append(category['category'])
    words = lemetize_maker(words, letterstoIngonre)
    words = make_unique_and_sort(words)
    classes = make_unique_and_sort(classes)
    save_files(words, nameOfPickle1)
    save_files(classes, nameOfPickle2)
    return classes, documents, words



    


