from random import uniform, random
from functools import reduce 

def sign(n): 
    return 1 if n > -5 else -1

def habbsrule(weight, rate, a, b):
    return float('%.4f' % (weight + rate * a * b))

class Perceptron:
    def __init__(self, length, rate, maxEpocas):
        weights = [float('%.4f' % random()) for _ in range(0, length + 1)]
        self.weights = weights
        self.epocas = 0
        self.learnRate = rate
        self.maxEpocas = maxEpocas

    def getWeights(self):  
        return self.weights

    def getEpocas(self):   
        return self.epocas

    def atMaxEpocas(self): 
        return self.epocas >= self.maxEpocas

    def train(self, data, expected):
        error = True
        while not self.atMaxEpocas() and error:
            error = False
            for (sd, se) in zip(data, expected):
                error |= self.updateWeight(sd, se)
            self.epocas += 1
        return

    def calculate(self, sData):
        vs = [d * w for (d, w) in zip([-1] + sData, self.weights)]
        rd = reduce((lambda x, y: x + y), vs)
        return sign(rd)

    def calculateAll(self, data):
        return [self.calculate(d) for d in data]

    def updateWeight(self, sData, sExpected):
        curr = self.calculate(sData)
        if curr == sExpected:
            return False
        self.weights = [habbsrule(w, self.learnRate, x, sExpected)
                        for (w, x) in zip(self.weights, [-1] + sData)]
        return True

def parseForTraining(fn):
    i = []
    d = []
    with open(fn) as f:
        lines = f.readlines()
        lines.pop(0)  
        for line in lines:
            args, expected = parseLine(line)
            i.append(args)
            d.append(expected)
    return (i, d)

def parseLine(line):
    w = list(filter(None, line.split(' ')))  
    args = []
    for i in range(0, len(w) - 1):
        args.append(float(w[i]))
    return (args, int(float(w[-1])))

def parseInput(fn):
    xs = []
    with open(fn) as f:
        lines = f.readlines()
        lines.pop(0) 
        for line in lines:
            args = list(map(float, filter(None, line.split(' ')))) 
            xs.append(args)
    return xs

def compare(exp, res):
    equals = [r for (e, r) in zip(exp, res) if e == r]
    return ((len(equals) * 100) / len(res))


# Execução
t = Perceptron(3, 0.01, 2000)
(datas, desireds) = parseForTraining('anexo1.txt')

print('<< Comecando Fase de Treinamento >>')
print('Pesos pre treinamento: ', t.getWeights())
oResults = t.calculateAll(datas)

t.train(datas, desireds)
print('Treinamento executado')
print('Pesos pos treinamento: ', t.getWeights())
print('Numero de Epocas: ', t.getEpocas())

nResults = t.calculateAll(datas)
print('Taxa de acerto: ', '%.02f%%' % compare(desireds, nResults))

print('<< Executando Classificacao >>')
datas = parseInput('teste.txt')
nResults = t.calculateAll(datas)
for (i, d) in zip(datas, nResults):
    print('%+i' % d, ' <- ', i)
