import warnings, logging, math, random, sklearn, logging, time, inspect
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics, svm
from sklearn.model_selection import KFold

class FWLS:
    '''
        Class for Feature Weighted Linear Stacking (FWLS) scheme for producing
        ensembles from a collection of heterogeneous models for multiclass
        classification.
    '''

    def __init__(self, dataset):
        self.Models = ["Logistic regression", "SVM", "Random forest"] # list of models
        self.D = dataset
        self.B = RandomForestClassifier #blenging algorithms
        self.P = [200,100,999] #preference array
        self.N = 20
        self.k = 10

    def genParams(self): # P-preference array; N-Number of models
        logging.info("  Generating HyperParameters")
        sample_dirichlets = np.random.dirichlet(self.P) # Sample a distribution from a Dirichlet with parameters P
        assert len(sample_dirichlets) == 3
        assert self.N >= len(self.P)
        No_of_models = []
        assert No_of_models != None
        for i in range(len(self.P)):
            No_of_models.append(int(math.floor(sample_dirichlets[i]* self.N))) # We Will generate Ni models from A
        assert sum(No_of_models) <= self.N
        HyperParameterMatrix = []

        for i in range(len(self.P)):
            for j in range(No_of_models[i]):
                m = []
                parameters = {}
                if(self.Models[i] == "Logistic regression"):
                    # print "Logistic regression"
                    pram_grid = {'C' :[0.001, 0.01, 0.1, 1, 10, 100, 1000]} # paramater C
                    parameters['pram_grid'] = pram_grid
                    solvers = ['newton-cg', 'lbfgs', 'liblinear'] # paramater solvers 'sag'
                    parameters['solvers'] = random.choice(solvers)
                    penalty = ['l1', 'l2'] # l2 only for liblinear # paramater penalty
                    if(parameters['solvers'] == 'liblinear'):
                        parameters['penalty'] = random.choice(penalty)
                    else:
                        parameters['penalty'] = 'l2'
                    m.append("Logistic Regression")
                    m.append(parameters)
                elif(self.Models[i] == "SVM"):
                    # print "SVM"
                    kernal = ['linear', 'poly', 'rbf', 'sigmoid']
                    parameters['kernal'] = random.choice(kernal)
                    parameters['gamma']  = [0.001, 0.01, 0.1, 1]
                    parameters['C'] = [0.001, 0.01, 0.1, 1, 10]
                    m.append("SVM")
                    m.append(parameters)
                elif(self.Models[i] == "Random forest"):
                    # print "Random Forest"
                    parameters['n_estimators'] = [10, 20, 50, 75]
                    parameters['max_features'] = random.choice(['auto', 'sqrt', 'log2'])
                    m.append("Random forest")
                    m.append(parameters)
                HyperParameterMatrix.append(m)
        parameters = {}
        B_psi = []
        parameters['n_estimators'] = 10
        parameters['max_features'] = 10
        B_psi.append("Random Forest")
        B_psi.append(parameters)
        return B_psi, No_of_models, HyperParameterMatrix

    def blend(self, Fi, L, B_psi, No_of_models, HyperParameterMatrix):
        p = 0.5
        Dfw = []
        Dfw_y = []
        _Mijs = []
        for i in range(L):
            D, _D, D_y, _D_y = train_test_split(Fi.data, Fi.target, test_size= p, random_state=0) # setting D amd DprimeComplement
            Total = 0
            logging.info("  Tuning Models...")
            for j in range(len(self.Models)):
                for s in range(No_of_models[j]):
                    if(self.Models[j] == "Logistic regression"):
                        clf = GridSearchCV(LogisticRegression(penalty=HyperParameterMatrix[Total][1]['penalty'], solver = HyperParameterMatrix[Total][1]['solvers']), HyperParameterMatrix[Total][1]['pram_grid'])
                        clf.fit(D, D_y)
                        _Mijs.append(clf)
                    elif(self.Models[j] == "SVM"):
                        param_grid = {'C': HyperParameterMatrix[Total][1]['C'], 'gamma' : HyperParameterMatrix[Total][1]['gamma']}
                        clf = GridSearchCV(svm.SVC(kernel=HyperParameterMatrix[Total][1]['kernal'], probability=True), param_grid)
                        clf.fit(D, D_y)
                        _Mijs.append(clf)
                    elif(self.Models[j] == "Random forest"):
                        param_grid = {'n_estimators' : HyperParameterMatrix[Total][1]['n_estimators'] }
                        clf = GridSearchCV(RandomForestClassifier(n_jobs=-1, max_features = HyperParameterMatrix[Total][1]['max_features']), param_grid)
                        clf.fit(D, D_y)
                        _Mijs.append(clf)
                    Total = Total + 1
            logging.info(" Create feature weighted Dataset")
            for data in range(len(_D)):
                row = []
                for k in range(len(_D[data])):
                    if L == 0:
                        row.extend([_D[data][k].tolist()])
                    Total = 0
                    for i in range(len(self.Models)):
                        for j in range(No_of_models[i]):
                            if(self.Models[i] == "Logistic regression"):
                                row.extend(_D[data][k]*_Mijs[Total].predict_proba(_D)[k])
                                row.extend(_Mijs[Total].predict_proba(_D)[k])
                            elif(self.Models[i] == "SVM"):
                                row.extend(_D[data][k]*_Mijs[Total].predict_proba(_D)[k])
                                row.extend(_Mijs[Total].predict_proba(_D)[k])
                            elif(self.Models[j] == "Random forest"):
                                row.extend(_D[data][k]*_Mijs[Total].predict_proba(_D)[k])
                                row.extend(_Mijs[Total].predict_proba(_D)[k])
                            Total = Total + 1
                Dfw.append(np.array(row))
                Dfw_y.append(_D_y[data])
        dataset = sklearn.datasets.base.Bunch(data=Dfw, target=Dfw_y)
        return dataset

    def blendModel(self, Fi, L, B_psi, No_of_models, HyperParameterMatrix):
        dataset = self.blend(Fi, L, B_psi, No_of_models, HyperParameterMatrix)
        logging.info("  Preparing the ensemble...")
        M_clf = self.B(n_jobs=-1, max_features = B_psi[1]['max_features'], n_estimators = B_psi[1]['n_estimators'])
        M_clf.fit(dataset.data, dataset.target)
        return M_clf

    def blendingEnsemble(self):
        L = 2
        Mm = []
        Errors_r = []
        for r in range(5): # value of R is 10 here iterations
            logging.info("  Iteration #" + str(r))
            B_psi, No_of_models, HyperParameterMatrix = self.genParams() # search for hyperparameters
            Mm.append([B_psi, No_of_models, HyperParameterMatrix])
            Errors = []

            kf = KFold(n_splits=10, shuffle=True)
            for Fi_index, _Fi_index in kf.split(self.D.data):
                Fi_data, _Fi_data = self.D.data[Fi_index], self.D.data[_Fi_index]
                Fi_data_y, _Fi_data_y = self.D.target[Fi_index], self.D.target[_Fi_index]

                Fi = sklearn.datasets.base.Bunch(data=Fi_data, target=Fi_data_y)
                _Fi = sklearn.datasets.base.Bunch(data=_Fi_data, target=_Fi_data_y)

                M = self.blendModel(Fi, L, B_psi, No_of_models, HyperParameterMatrix)
                dataset = self.blend(_Fi, L, B_psi, No_of_models, HyperParameterMatrix)
                logging.info(" " + str(metrics.accuracy_score(dataset.target,M.predict(dataset.data))))
                Errors.append(metrics.accuracy_score(dataset.target,M.predict(dataset.data)))
            logging.info("  Average #"  + str(r) + " " + str(sum(Errors)/len(Errors)))
            Errors_r.append(sum(Errors)/len(Errors))
        logging.info("  Final " + str(np.array(Errors_r)))
        print np.argmax(np.array(Errors_r)) # since we are using accuracy_score instead of error
        index = np.argmax(np.array(Errors_r))
        MM = self.blendModel(self.D,L,Mm[index][0], Mm[index][1], Mm[index][2])
        logging.info("  Final ensemble completed")
        return MM

    def worker(self):
        logging.info("  Starting the FWLS procedure")
        func = inspect.currentframe().f_back.f_code
        logging.debug(" %s: %s in %s:%i" % (
                    "message",
                    func.co_name,
                    func.co_filename,
                    func.co_firstlineno
                ))
        self.blendingEnsemble()
        pass

if __name__ == '__main__':
    logdatetime = time.strftime("%m%d")
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                        filename='./FLWS' + logdatetime + '.log',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.DEBUG)
    D = datasets.load_iris()
    f = FWLS(D)
    Model = f.worker()
    logging.info("Procedure terminated without errors")
