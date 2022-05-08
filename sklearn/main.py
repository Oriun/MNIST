import os
import time
import json
from dataset import Dataset
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

start = int(time.time())
prefix = 'scores/'+str(start)+'/'

# n_neighbors = 5 # default
# weights = 'uniform' # default

# Give better results
n_neighbors = 3
weights = 'distance'

training = Dataset(True, print=True)
test = Dataset(print=True)
print('Datasets loaded')


print('KNN launch')
knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier

print('\nStart KNN Fitting')
knn_classifier.fit(training.reshape(), training.labels)
print('\nStart KNN Prediction')
knn_predictions = knn_classifier.predict(test.reshape())
print('\n\nPrediction ended')
print('Duration '+str(time.time() - start)+'s')


########### ABOVE HERE IS ONLY TO PROCESS RESULTS, SAVE AND MAKE CHARTS
########### Results are stored in the scores directory

os.makedirs(os.path.dirname('scores/'), exist_ok=True)
os.makedirs(os.path.dirname(prefix+'/'), exist_ok=True)
knn_predictions.tofile(prefix+'/knn_predictions-' +
                       str(int(start))+'.txt', sep=';')

total = len(test.labels)
misclassified = 0

keys = list(map(str, test.distribution.keys()))

results = dict(
    zip(keys, [dict(zip(keys, [0 for _ in keys])) for __ in keys]))


for i in range(len(knn_predictions)):
    res = str(knn_predictions[i])
    expected = str(test.labels[i])
    if res != expected:
        misclassified += 1
    results[res][expected] += 1

succes_rate = str(1 - (misclassified / total))
print('Success Rate of '+succes_rate+'%')
print('\n\nSaving Results to '+ prefix)


with open(prefix+'/scores-'+str(start)+'.json', 'w') as outfile:
    json.dump({
        'result': results,
        'total': total,
        'misclassified': misclassified,
        'n_neighbors': n_neighbors,
        'weights': weights
    }, outfile, indent=4)

confusion_matrix = list(map(lambda val: list(val.values()), results.values()))

plt.title("Matrice de confusion, taux de succès à "+succes_rate+"%")

ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.box(on=None)
plt.subplots_adjust(left=0.2, bottom=0.2)

table = plt.table(
    cellText=confusion_matrix,
    colLabels=list(map(lambda x: ' '+x+' ', keys)),
    rowLabels=keys,
    loc='center'
)

table.scale(1, 1.5)


plt.savefig(prefix+'/graph-'+str(start)+'.png')
plt.show()
