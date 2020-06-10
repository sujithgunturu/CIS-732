import json
#read the prepaded data file
intents_file = open('data.json').read()
#get class labels from the dataset
classesTotrain = json.loads(intents_file)
#few letters to igone can add more
# letters_to_ignore = ['!', '?', ',', '.','`','@', '#', '^', '*', "(" ")"]
letters_to_ignore = []
from train import pre_process, create_X_Y, compile_fit, plot_accuracyVSepochs
#creating list of tuples
classes, documents, words = pre_process(classesTotrain, letters_to_ignore, "words", "classes")
#creating Data
X, Y = create_X_Y(classes, documents, words)
# train test split
from sklearn.model_selection import  train_test_split
train_x, test_x, train_y, test_y = train_test_split( X, Y, test_size=0.25, random_state=42)
training_board, model = compile_fit(train_x, train_y)
plot_accuracyVSepochs(training_board)
import numpy as np
from sklearn.metrics import classification_report
pred1=model.predict(np.array(test_x))
print(classification_report(np.array(test_y),pred1.round()))

#https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
#reference : sklearn stable version one vs rest classifier
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt



#performing tests on thetesting data
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
#reference : sklearn stable version one vs rest classifier
x = np.array(test_x)
y = np.array(test_y)
cdash = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 ))
y = label_binarize(y, classes=np.arange(len(train_y[0])))
y_score = cdash.fit(np.array(train_x), np.array(train_y)).decision_function(x)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(train_y[0])):
    fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ')
plt.legend(loc="lower right")
plt.show()



#ROC curve by using np.array().ravel which transforms the array into 1d
#
from sklearn.metrics import confusion_matrix
y_pred_keras = model.predict(x)
print(y_pred_keras.shape)
score = roc_auc_score(y, y_pred_keras)
print(score)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y.ravel(), y_pred_keras.ravel())
auc_keras = auc(fpr_keras, tpr_keras)

cm = confusion_matrix(y.argmax(axis=1), y_pred_keras.argmax(axis=1))
print(cm)
print("keras area under the curve", auc_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()



