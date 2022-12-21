import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")


class pointstructure_kernel:
    def __init__(self, traindata, trainresult, C, toler,kernel):
        self.data = traindata
        self.label = trainresult
        self.C = C
        self.tol = toler
        self.len = len(traindata)
        self.alphas = np.mat(np.zeros((len(traindata), 1)))
        self.b = 0
        self.ecache = np.mat(np.zeros((len(traindata), 2)))  # It is used to indicate whether the point is selected and what is the calculation error
        self.K = np.mat(np.zeros((self.len, self.len)))
        for i in range(self.len):
            self.K[:, i] = kernelTrans(self.data, self.data[i], kernel)


class pointstructure:
    def __init__(self, traindata, trainresult, C, toler):
        self.data = traindata
        self.label = trainresult
        self.C = C
        self.tol = toler
        self.len = len(traindata)
        self.alphas = np.mat(np.zeros((len(traindata), 1)))
        self.b = 0
        self.ecache = np.mat(np.zeros((len(traindata), 2)))  # It is used to indicate whether the point is selected and what is the calculation error


def kernelTrans(traindata, data_line, kernel):
    """
    kernel
    Args:
        traindata
        data_line
        kernel

    Returns:

    """
    m, n = np.shape(traindata)
    K = np.mat(np.zeros((m, 1)))
    if kernel[0] == 'lin':
        K = traindata * data_line.T
    elif kernel[0] == 'rbf':
        for j in range(m):
            deltaRow = traindata[j, :] - data_line
            K[j] = deltaRow * deltaRow.T
        # 径向基函数的高斯版本
        K = np.exp(K / (-1 * kernel[1] ** 2))
    return K


def change_Alpha(aj, H, L):
    """
    Adjust the value of aj so that aj is at L<=aj<=H
    Args:
        aj  target value
        H   max value
        L   min value
    Returns:
        aj  target value
    """
    aj = min(aj, H)
    aj = max(L, aj)
    return aj


def calculate_EK(point, k):
    """
    Args:
        point
        k   A specific line

    Returns:
        Ek  The prediction result is compared with the real result, and the calculation error Ek
    """
    forcast = np.multiply(point.alphas, point.label).T * (point.data * point.data[k].T) + point.b  # w = Σ(1~n) a[n]*lable[n]*data[n]
    Ek = forcast - float(point.label[k])
    return Ek


def updateEk(point, k):
    """
    Calculate the error value and store it in the cache
    Args:
        point
        k   Row number of a column
    """
    Ek = calculate_EK(point, k)  # Error calculation: the difference between the predicted value and the true value
    point.ecache[k] = [1, Ek]


def select_anotherpoint(i, point, Ei):
    """
    Select the appropriate second alpha value to ensure that the maximum step size is used in each optimization.
    The error of this function is related to the first alpha value Ei and subscript i.
    Args:
        i   Specific line
        point
        Ei  The prediction result is compared with the real result, and the calculation error Ei

    Returns:
        j  Randomly selected row j
        Ej The prediction result is compared with the real result, and the calculation error Ej
    """
    max_number = -1
    max_delta_E = 0
    Ej = 0
    point.ecache[i] = [1, Ei]  # First, set the input value Ei to be valid in the cache. The validity here means that it has been calculated.
    validEcacheList = np.nonzero(point.ecache[:, 0].A)[0]

    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   # Loop over all values and select the value that causes the maximum change
            if k == i:
                continue

            # Calculate Ek error: the difference between predicted value and true value
            Ek = calculate_EK(point, k)
            deltaE = abs(Ei - Ek)
            if deltaE > max_delta_E:
                max_number = k
                max_delta_E = deltaE
                Ej = Ek
        return max_number, Ej

    else:  # If it is the first cycle, randomly select an alpha value
        j = i
        while j == i:
            j = np.random.randint(0, point.len - 1)
        Ej = calculate_EK(point, j)   # Calculate Ek error: the difference between predicted value and true value
    return j, Ej


def searchpointpair(i, point):
    """
    Args:
        i   A specific line
        point

    Returns:
        0   No optimal value found
        1   The optimal value is found, and point.cache is put into the cache
    """

    Ei = calculate_EK(point, i)   # Calculate Ek error: the difference between predicted value and true value

    if ((point.label[i] * Ei < -point.tol) and (point.alphas[i] < point.C)) or ((point.label[i] * Ei > point.tol) and (point.alphas[i] > 0)):
        j, Ej = select_anotherpoint(i, point, Ei)  # Select j corresponding to the maximum error for optimization. The effect is more obvious
        alpha_I_old = point.alphas[i].copy()
        alpha_J_old = point.alphas[j].copy()

        #L and H are used to adjust alpha [j] between 0-C. If L==H, return 0 without any change
        if point.label[i] != point.label[j]:
            L = max(0, point.alphas[j] - point.alphas[i])
            H = min(point.C, point.C + point.alphas[j] - point.alphas[i])
        else:
            L = max(0, point.alphas[j] + point.alphas[i] - point.C)
            H = min(point.C, point.alphas[j] + point.alphas[i])
        if L == H:
            return 0

        # Update alpha, and eheta is the optimal modification amount of alpha [j]. If eheta==0, you need to exit the current iteration process of the for loop
        eheta = 2.0 * point.data[i] * point.data[j].T - point.data[i] * point.data[i].T - point.data[j] * point.data[j].T  # -x1*x1-x2*x2+2*x1*x2

        point.alphas[j] -= point.label[j] * (Ei - Ej) / eheta     # Calculate a new alphas [j] value
        point.alphas[j] = change_Alpha(point.alphas[j], H, L)    # The value of aj is rounded so that aj is at L<=aj<=H
        updateEk(point, j)  # Update error buffer, add j

        if (abs(point.alphas[j] - alpha_J_old) < 0.00001):    # Check whether alpha [j] is only a slight change. If so, exit the for loop.
            return 0

        point.alphas[i] += point.label[i]*(Ei - Ej) / eheta   # Then alpha [i] also changes in the opposite direction
        point.alphas[i] = change_Alpha(point.alphas[i], H, L)  # Adjust the value of ai so that ai is at L<=ai<=H
        updateEk(point, i)  # Update Error Cache

        # Update the value of b
        b1 = point.b - Ei - point.label[i] * (point.alphas[i] - alpha_I_old) * point.data[i] * point.data[i].T - point.label[j] * (point.alphas[j] - alpha_J_old) * point.data[i] * point.data[j].T
        b2 = point.b - Ej - point.label[i] * (point.alphas[i] - alpha_I_old) * point.data[i] * point.data[j].T - point.label[j] * (point.alphas[j] - alpha_J_old) * point.data[j] * point.data[j].T
        if (0 < point.alphas[i]) and (point.C > point.alphas[i]):
            point.b = b1
        elif (0 < point.alphas[j]) and (point.C > point.alphas[j]):
            point.b = b2
        else:
            point.b = (b1 + b2) / 2
        return 1
    else:
        return 0


def smop(traindata,trainresult, C, toler, maxcount):
    """
    Args:
        dataMatIn
        classLabels
        C
        toler
        maxcount
    Returns:
        b
        alphas
    """

    point = pointstructure(np.mat(traindata), np.mat(trainresult).transpose(), C, toler)    # Create a struct object
    count = 0
    pointpair = True     # When notpointpair=true, it indicates that there is an alpha pair, or it enters the loop for the first time
    countalphaPairsChange = 0   # Number of key value pairs for optimization

    # Exit if the alphaPairs still remain unchanged after the loop iteration ends or after the loop traverses all the alphas
    while (count < maxcount) and ((countalphaPairsChange > 0) or (pointpair)):    # One is that the cycle times are smaller than maxIter times, and there are point pairs that can be optimized; the other is that the first time
        countalphaPairsChange = 0
        if pointpair:
            for i in range(len(traindata)):   # Traverse all possible alphas on the dataset
                countalphaPairsChange += searchpointpair(i, point)   # If there is an alpha pair,+1 exists
            count += 1

        else:
            nonBoundIs = np.nonzero((point.alphas.A > 0) * (point.alphas.A < C))[0]   # . A is converted into an array, and all non boundary alpha values are traversed, that is, the values not on the boundary 0 or C.
            for i in nonBoundIs:
                countalphaPairsChange += searchpointpair(i, point)
            count += 1

        if pointpair:
            pointpair = False
        elif countalphaPairsChange == 0:    # If there is no point pair, it will traverse for the last time. If there is no point pair, it will exit the loop.
            pointpair = True
    return point.b, point.alphas


def smop_kernel(traindata,trainresult, C, toler, maxcount, kernel):
    """
        Args:
            dataMatIn
            classLabels
            C
            toler
            maxcount
        Returns:
            b
            alphas
        """

    point = pointstructure_kernel(np.mat(traindata), np.mat(trainresult).transpose(), C, toler, kernel)
    count = 0       # 循环次数
    pointpair = True     # 当notpointpair=true 表示有alpha对，或第一次进入循环
    countalphaPairsChange = 0   # 可供优化的键值对个数

    # Exit if the alphaPairs still remain unchanged after the loop iteration ends or after the loop traverses all the alphas
    while (count < maxcount) and ((countalphaPairsChange > 0) or (
    pointpair)):  # One is that the cycle times are smaller than maxIter times, and there are point pairs that can be optimized; the other is that the first time
        countalphaPairsChange = 0
        if pointpair:
            for i in range(len(traindata)):  # Traverse all possible alphas on the dataset
                countalphaPairsChange += searchpointpair(i, point)  # If there is an alpha pair,+1 exists
            count += 1

        else:
            nonBoundIs = np.nonzero((point.alphas.A > 0) * (point.alphas.A < C))[0]  # . A is converted into an array, and all non boundary alpha values are traversed, that is, the values not on the boundary 0 or C.
            for i in nonBoundIs:
                countalphaPairsChange += searchpointpair(i, point)
            count += 1

        if pointpair:
            pointpair = False
        elif countalphaPairsChange == 0:  # If there is no point pair, it will traverse for the last time. If there is no point pair, it will exit the loop.
            pointpair = True
    return point.b, point.alphas


def calculate_ws(alphas, dataArr, classLabels):
    """
    Calculate w value based on alpha
    Args:
        alphas        Lagrange multiplier
        dataArr
        classLabels

    Returns:
        wc  regression coefficient
    """
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i].T)
    return w

def split_train_test_kernel(data,k):
    indices = np.random.permutation(len(data))
    data = data.iloc[indices]

    result_all = []
    result_test = []
    acc_all = []
    pre_all = []
    recall_all = []
    F1score_all = []
    count = 0

    for i in range(k):
        if i==0 :
            test_set = data[:int((i+1)/k*len(data))]
            train_set = data[int((i+1)/k*len(data)):]
        else:
            test_set = data[int(i/k * len(data)):int((i + 1) / k * len(data))]
            train_set1 = data[:int(i/k * len(data))]
            train_set2=data[int((i + 1) / k * len(data)):]
            train_set=train_set1.append(train_set2)
        train_data=train_set.iloc[:,:-1]
        train_result=train_set.iloc[:,-1]
        test_data = test_set.iloc[:, :-1]
        test_result = test_set.iloc[:, -1]

        train_data = np.array(train_data)
        train_data = train_data.tolist()

        train_result = np.array(train_result)
        train_result = train_result.tolist()

        test_data = np.array(test_data)
        test_data = test_data.tolist()

        test_result = np.array(test_result)
        test_result = test_result.tolist()

        kernel = ('lin')
        b, alphas=smop_kernel(train_data, train_result, 1, 0.00001, 40, kernel)

        new_trian_data = np.mat(train_data)
        label = np.mat(train_result).transpose()
        nonezero_number = np.nonzero(alphas.A > 0)[0]
        train_data_nonezero = new_trian_data[nonezero_number]
        label_nonezero = label[nonezero_number]
        new_test_data = np.mat(test_data)
        m, n = np.shape(new_test_data)
        for i in range(m):
            kernelEval = kernelTrans(train_data_nonezero, new_test_data[i, :], kernel)
            predict = kernelEval.T * np.multiply(label_nonezero, alphas[nonezero_number]) + b
            result_all.append(np.sign(predict))
            result_test.append(test_result[i])
    for i in range(len(result_all)):
        if result_all[i] == result_test[i]:
            count += 1
    for i in range(2):
        TP=FP=FN=TN=0
        if i == 0:
            i = -1
        for k in range(len(result_all)):
            if result_all[k] == i and result_test[k] == i:
                    TP += 1
            if result_all[k] == i and result_test[k] != i:
                    FP += 1
            if result_all[k] != i and result_test[k] == i:
                    FN += 1
            if result_all[k] != i and result_test[k] != i:
                    TN += 1
        acc = (TP+TN)/(TP+TN+FP+FN)
        pre = TP/(TP+FP)
        recall=TP/(TP+FN)
        F1score = 2*pre*recall / (pre + recall)
        acc_all.append(acc)
        pre_all.append(pre)
        recall_all.append(recall)
        F1score_all.append(F1score)
    acc_count = 0
    pre_all_count = 0
    recall_all_count = 0
    F1score_all_count = 0
    for i in range(len(acc_all)):
        acc_count = acc_all[i] + acc_count
        pre_all_count = pre_all[i] + pre_all_count
        recall_all_count = recall_all[i] + recall_all_count
        F1score_all_count = F1score_all[i] + F1score_all_count
    acc_count = acc_count / (len(acc_all))
    pre_all_count = pre_all_count / (len(pre_all))
    recall_all_count = recall_all_count / len(recall_all)
    F1score_all_count = F1score_all_count / len(F1score_all)
    print("Accuracy：" + str(acc_count)
          + '\n' + "Precision：" + str(pre_all_count) + '\n' + "Recall：" + str(recall_all_count) + '\n'
          + "F1-Value：" + str(F1score_all_count) + '\n')


def split_train_test(data,k):

    indices = np.random.permutation(len(data))
    data=data.iloc[indices]

    result_all=[]
    result_test=[]
    acc_all=[]
    pre_all=[]
    recall_all=[]
    F1score_all=[]
    count = 0

    for i in range(k):
        if i==0 :
            test_set = data[:int((i+1)/k*len(data))]
            train_set = data[int((i+1)/k*len(data)):]
        else:
            test_set = data[int(i/k * len(data)):int((i + 1) / k * len(data))]
            train_set1 = data[:int(i/k * len(data))]
            train_set2=data[int((i + 1) / k * len(data)):]
            train_set=pd.concat([train_set1,train_set2])
        train_data=train_set.iloc[:,:-1]
        train_result=train_set.iloc[:,-1]
        test_data = test_set.iloc[:, :-1]
        test_result = test_set.iloc[:, -1]

        train_data = np.array(train_data)
        train_data = train_data.tolist()

        train_result = np.array(train_result)
        train_result = train_result.tolist()

        test_data = np.array(test_data)
        test_data = test_data.tolist()

        test_result = np.array(test_result)
        test_result = test_result.tolist()

        b, alphas=smop(train_data, train_result, 0.96, 0.001, 20)
        ws = calculate_ws(alphas, train_data, train_result)
        for item in test_data:
            result=item*np.mat(ws)+b
            result_all.append(np.sign(result))
        for i in range(len(test_result)):
            result_test.append(np.sign(test_result[i]))
    for i in range(len(result_all)):
        if result_all[i]==result_test[i]:
            count += 1
    for i in range(2):
        TP=FP=FN=TN=0
        if i == 0:
            i = -1
        for k in range(len(result_all)):
            if result_all[k] == i and result_test[k] == i:
                    TP += 1
            if result_all[k] == i and result_test[k] != i:
                    FP += 1
            if result_all[k] != i and result_test[k] == i:
                    FN += 1
            if result_all[k] != i and result_test[k] != i:
                    TN += 1
        acc = (TP+TN)/(TP+TN+FP+FN)
        pre = TP/(TP+FP)
        recall=TP/(TP+FN)
        F1score = 2*pre*recall / (pre + recall)
        acc_all.append(acc)
        pre_all.append(pre)
        recall_all.append(recall)
        F1score_all.append(F1score)
    acc_count =0
    pre_all_count =0
    recall_all_count =0
    F1score_all_count =0
    for i in range(len(acc_all)):
        acc_count = acc_all[i] + acc_count
        pre_all_count = pre_all[i]+pre_all_count
        recall_all_count = recall_all[i]+recall_all_count
        F1score_all_count = F1score_all[i] +F1score_all_count
    acc_count = acc_count/(len(acc_all))
    pre_all_count = pre_all_count/(len(pre_all))
    recall_all_count = recall_all_count/len(recall_all)
    F1score_all_count = F1score_all_count/len(F1score_all)
    print( "Accuracy：" + str(acc_count)
          + '\n' + "Precision：" + str(pre_all_count) + '\n' + "Recall：" + str(recall_all_count) + '\n'
          + "F1-Value：" + str(F1score_all_count) + '\n')


def normalization(df,result_count):
    properation1=result_count.iloc[0]/(result_count.iloc[0]+result_count.iloc[1])
    properation2=result_count.iloc[1]/(result_count.iloc[0]+result_count.iloc[1])

    newDataFrame = pd.DataFrame(index=df.index)
    columns = df.columns.tolist()
    for c in columns:
        d = df[c]
        MAX = d.max()
        MIN = d.min()
        newDataFrame[c] = ((d - MIN) / (MAX - MIN)).tolist()
    for index, item in newDataFrame.iterrows() :
        if item['Additional_attention']==0:
            newDataFrame.loc[index] = item/properation1
        else:
            newDataFrame.loc[index] = item/properation2
    return newDataFrame


if __name__ == '__main__':
    data=pd.read_csv("heart_failure_clinical_records_dataset.csv")

    df_coor = data.corr()
    fig, ax = plt.subplots(figsize=(6, 6),facecolor='w')
    sns.heatmap(data.corr(),annot=True, vmax=1, square=True, cmap="Blues", fmt='.2g')
    plt.title('correlation analysis')
    plt.show()

    weights = np.ones_like(data['age'])/float(len(data['age']))
    plt.hist(data['age'], rwidth=0.9, weights=weights)
    plt.title('age distribution')
    plt.xlabel('age')
    plt.ylabel('Probability')
    plt.show()

    weights = np.ones_like(data['time'])/float(len(data['time']))
    plt.hist(data['time'], rwidth=0.9, weights=weights)
    plt.title('time distribution')
    plt.xlabel('time')
    plt.ylabel('Probability')
    plt.show()

    weights=np.ones_like(data['serum_creatinine'])/float(len(data['serum_creatinine']))
    plt.hist(data['serum_creatinine'], rwidth=0.9, weights=weights)
    plt.title('serum_creatinine distribution')
    plt.xlabel('serum_creatinine')
    plt.ylabel('Probability')
    plt.show()

    result_count=pd.value_counts(data["Additional_attention"])   # Count the number of each category
    data=normalization(data,result_count)   # Normalized the data, and coefficients are assigned to each category of data to make the results more accurate
    for i in range(len(data)):
        if data["Additional_attention"].iloc[i]==0:
            data["Additional_attention"].iloc[i] =-1
        else:
            data["Additional_attention"].iloc[i] = 1
    for j in range(10):
        split_train_test(data, 10)
        j += 1
