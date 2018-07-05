import datetime as dt
import matplotlib.pyplot as plt
import sklearn.svm as svm
import calendar


def mean_accuracy_with_confidence_interval(predict_array, interval, Y_test):
    accuracy = 0
    for hour in range(0,len(predict_array)):
        if (predict_array[hour] + interval >= Y_test[hour]) and (predict_array[hour] - interval <= Y_test[hour]):
            accuracy += 1
    return accuracy / 24


def compute_prediction(values, dates, consumptions, predicted):
    for element in values:
        x_vector = []
        temp = element.split(",")
        date = dt.datetime.fromtimestamp(int(temp[0]) / 1000)
        consumption = float(temp[1])

        if (calendar.weekday(date.year, date.month, date.day) < 5):
            x_vector.append(1)
        else:
            x_vector.append(0)

        if len(consumptions) > 48:
            x_vector += [consumptions[-i] for i in range(1, 49)]
        elif len(consumptions) == 0:
            x_vector += [5 for _ in range(len(consumptions), 48)]
        else:
            x_vector += [consumptions[-i] for i in range(1, len(consumptions))]
            x_vector += [5 for _ in range(len(consumptions), 49)]

        if (date.year == 2018 and date.month == 5 and date.day == 31):
            X_test[date.hour].append(x_vector)
            Y_test[date.hour].append(consumption)
        else:
            X_train[date.hour].append(x_vector)
            Y_train[date.hour].append(consumption)

        dates += [date]
        consumptions += [consumption]
    print("From {0} to {1}".format(dates[0], dates[-1]))
    # now i create one svm for each hour
    for hour in X_train:
        SVR_model = svm.SVR(kernel='rbf', C=3e21, gamma=3.5e21).fit(X_train[hour], Y_train[hour])
        predicted += [SVR_model.predict(X_test[hour])]
        print("{0} value: {1}".format(hour, predicted[hour]))



file = open("../consumptions.txt") # tutti i consumi dal 1/6/2017 al 1/6/2018 alle due
line = file.readline().strip()
building = line.split("\t")[0]
values = line.split("\t")[1].split(" ")
dates = []
consumptions = []
X_train = {}
X_test = {}
Y_train = {}
Y_test = {}
predict_y_array = []
for i in range(0,24):
    X_train[i] = []
    X_test[i] = []
    Y_test[i] = []
    Y_train[i] = []

compute_prediction(values, dates, consumptions, predict_y_array)
accuracy = mean_accuracy_with_confidence_interval(predict_y_array, interval=1e5, Y_test=Y_test)

print("Mean accuracy: " + str(accuracy))
plt.plot(dates[-26:-2],consumptions[-26:-2], color='red')
plt.plot(dates[-26:-2], predict_y_array)
plt.xlabel('dates')
plt.ylabel('Kwh')
plt.gcf().autofmt_xdate()
plt.show()

#TODO aggiungere temperature al vettore di input, riallenarsi sul giorno successivo, prendere i dati da internet.