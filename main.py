from sklearn.datasets import fetch_openml
from sklearn import svm 

import sys
import os.path

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == 'retrain':
            from joblib import dump

            mnist = fetch_openml('MNIST_784')
            X, y = mnist.data, mnist.target

            clf = svm.SVC(verbose=True)
            clf.fit(X, y)

            dump(clf, 'model')
    
    if os.path.isfile('model') == False:
        print('err: "model" does not exist')
        print('\tplease run with "retrain" arguement')
        return;
    

if __name__ == '__main__':
    main()
    print('exited...')
