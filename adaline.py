from random import random
from functools import reduce

class Adaline:
    def __init__(self, length, rate, max_epochs):
        self.weights = [random() for _ in range(length + 1)]
        self.learn_rate = rate
        self.max_epochs = max_epochs
        self.epochs = 0

    def get_weights(self):
        return self.weights

    def get_epochs(self):
        return self.epochs

    def at_max_epochs(self):
        return self.epochs >= self.max_epochs

    def train(self, data, expected):
        for _ in range(self.max_epochs):
            total_error = 0
            for s_data, s_expected in zip(data, expected):
                output = self.calculate(s_data, activation=False)
                error = s_expected - output
                total_error += error ** 2
                # Atualiza os pesos usando o erro linear (não usando a função de ativação)
                self.weights = [w + self.learn_rate * error * x for w, x in zip(self.weights, [-1] + s_data)]
            self.epochs += 1
            if total_error < 9.3:  
                break

    def calculate(self, s_data, activation=True):
        vs = [d * w for d, w in zip([-1] + s_data, self.weights)]
        result = sum(vs)
        # Usar uma função de ativação linear (ou seja, a própria saída) se activation=False
        return 1 if result >= 0 else -1 if activation else result

    def calculate_all(self, data):
        return [self.calculate(d) for d in data]


def parse_for_training(fn):
    inputs, expected = [], []
    with open(fn) as f:
        lines = f.readlines()
        lines.pop(0) 
        for line in lines:
            args, exp = parse_line(line)
            inputs.append(args)
            expected.append(exp)
    return inputs, expected

def parse_line(line):
    w = list(filter(None, line.split()))
    args = [float(w[i]) for i in range(len(w) - 1)]
    return args, int(float(w[-1]))

def parse_input(fn):
    data = []
    with open(fn) as f:
        lines = f.readlines()
        lines.pop(0)  
        for line in lines:
            args = list(map(float, filter(None, line.split())))
            data.append(args)
    return data

def compare(expected, results):
    matches = [1 for e, r in zip(expected, results) if e == r]
    return (len(matches) * 100) / len(results)


adaline = Adaline(3, 0.01, 2000)
datas, desireds = parse_for_training('anexo1.txt')

print('<< Iniciando Treinamento >>')
print('Pesos antes do treinamento: ', adaline.get_weights())

adaline.train(datas, desireds)
print('Treinamento concluído')
print('Pesos após o treinamento: ', adaline.get_weights())
print('Número de épocas: ', adaline.get_epochs())

n_results = adaline.calculate_all(datas)
print('Taxa de acerto: ', '%.02f%%' % compare(desireds, n_results))

print('<< Executando Classificação >>')
datas = parse_input('teste.txt')
n_results = adaline.calculate_all(datas)
for i, d in zip(datas, n_results):
    print(f'{d:+d} <- {i}')