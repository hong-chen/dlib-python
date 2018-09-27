import dlib
try:
    import cPickle as pickle
except ImportError:
    import pickle

x = dlib.vectors()
y = dlib.array()

x.append(dlib.vector([1, 2, 3, -1, -2, -3]))
y.append(+1)

x.append(dlib.vector([-1, -2, -3, 1, 2, 3]))
y.append(-1)

# SVM is Support Vector Machine
svm = dlib.svm_c_trainer_linear()
# svm = dlib.svm_c_trainer_radial_basis()
# svm = dlib.svm_c_trainer_radial_hisogram_intersection()
svm.be_verbose()
svm.set_c(10)

classifier = svm.train(x, y)

print("prediction for first sample:  {}".format(classifier(x[0])))
print("prediction for second sample: {}".format(classifier(x[1])))

with open('saved_model.pickle', 'wb') as handle:
    pickle.dump(classifier, handle, 2)


# print(x)
# print(y)

