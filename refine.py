import sys
sys.path.append('/home/zhoujianyu/project/rnaopen/lib/lib/python3.8/site-packages')
import numpy as np

data_dir = 'tasks/rl'

def compare(filea, fileb):
    a = []
    b = []
    for i in range(1, 113):
        with open(data_dir + '/' + filea + '/' + str(i) + '.predict.txt', 'r') as f:
            tmp = []
            for line in f:
                tmp.append(float(line.strip()))
        a.append(tmp)

    for i in range(1, 113):
        with open(data_dir + '/' + fileb + '/' + str(i) + '.predict.txt', 'r') as f:
            tmp = []
            for line in f:
                tmp.append(float(line.strip()))
        b.append(tmp)

    res = []
    for i in range(0, 112):
        rsmd = np.sqrt(np.mean((np.array(a[i]) - np.array(b[i]))**2))
        res.append(rsmd)
        if(rsmd > 0.3):
            print(i+1, '\t', len(a[i]) , '\t', rsmd)
    print('!!!', np.mean(res))

def generate(filea, fileb, rate):
    a = []
    b = []
    for i in range(1, 113):
        with open(data_dir + '/' + filea + '/' + str(i) + '.predict.txt', 'r') as f:
            tmp = []
            for line in f:
                tmp.append(float(line.strip()))
        a.append(tmp)

    for i in range(1, 113):
        with open(data_dir + '/' + fileb + '/' + str(i) + '.predict.txt', 'r') as f:
            tmp = []
            for line in f:
                tmp.append(float(line.strip()))
        b.append(tmp)

    res = []
    
    for i in range(0, 112):
        with open(data_dir + '/final_trial/' + str(i+1) + '.predict.txt', 'w') as f:
            rsmd = np.sqrt(np.mean((np.array(a[i]) - np.array(b[i]))**2))
            res.append(rsmd)
            if(rsmd > 0.3):
                print(i+1, '\t', len(a[i]) , '\t', rsmd)
                for t in range(len(a[i])):
                    print(a[i][t] * rate + b[i][t] * (1-rate), file = f)
            else:
                for t in range(len(a[i])):
                    print(b[i][t], file = f)
    print('!!!', np.mean(res))


compare('LinearFold', 'res19_0_49')
generate('LinearFold', 'res19_0_49', 0.3)
#compare('final_trial', 'res19_0_49')
#compare('final_trial', 'LinearFold')
#compare('final_trial', 'uniform')
#compare('LinearFold', 'uniform')
#compare('res19_0_49', 'uniform')
