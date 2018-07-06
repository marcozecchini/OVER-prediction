import datetime as dt
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import calendar

def mean_accuracy_with_confidence_interval(predict_array, interval_size, Y_test):
    accuracy = 0
    for hour in range(0,len(predict_array)):
        value_interval = predict_array[hour] * interval_size
        if (predict_array[hour] + value_interval >= Y_test[hour]) and (predict_array[hour] - value_interval <= Y_test[hour]):
            accuracy += 1
    return accuracy / 24

# the paper suggests: "they are substituted by the mean value of two neighbor points"
def overfit_filtering(consumption, consumptions, index):
    sum = 0
    sum += consumptions[index+1] if len(consumptions) > index+1 and consumptions[index+1] != 0.0 else consumption
    sum += consumptions[index-1] if len(consumptions) > index-1 and consumptions[index-1] != 0.0 else consumption
    return sum / 2


def avarage_consumption_last24(consumptions, consumption):
    sum = 0
    for hour in range(1,25):
        sum += consumptions[-hour] if len(consumptions) > hour and consumptions[hour] != 0.0 else overfit_filtering(consumption, consumptions, hour)
    return sum /24


def prepare_model_and_predict(values, dates, consumptions, predicted):
    for element in values:
        x_vector = []
        temp = element.split(",")
        date = dt.datetime.fromtimestamp(int(temp[0]) / 1000)
        consumption = float(temp[1])

        #timestamp
        x_vector += [int(temp[0]) / 1000]

        # TODO MANCA LA TEMPERATURA

        #weekday dummy variables filling
        for day in range(0,7):
            weekday = calendar.weekday(date.year, date.month, date.day)
            if (weekday == day):
                x_vector += [1]
                continue
            x_vector += [0]

        #consumption of the same hour 24 hour before
        x_vector += [consumptions[-24] if len(consumptions) > 24 else overfit_filtering(consumption, consumptions, 24)]

        #consumption at the same hour one week before
        x_vector += [consumptions[-24] if len(consumptions) > 168 else consumptions[-1] if len(consumptions) > 1 else overfit_filtering(consumption, consumptions, 168)]

        #consumption three hours before
        x_vector += [consumptions[-3] if len(consumptions) > 3 else overfit_filtering(consumption, consumptions, 3)]

        #consumption two hours before
        x_vector += [consumptions[-2] if len(consumptions) > 2 else overfit_filtering(consumption, consumptions, 2)]

        #consumption the hour before
        x_vector += [consumptions[-1] if len(consumptions) > 1 else overfit_filtering(consumption, consumptions, 1)]

        #avagare consumption of the last 24h hour
        x_vector += [avarage_consumption_last24(consumptions, consumption)]

        #filling or the train set or the test set
        if (date.year == 2018 and date.month == 5 and date.day == 29):
            X_test[date.hour].append(x_vector)
            Y_test[date.hour].append(consumption)
        else:
            X_train[date.hour].append(x_vector)
            Y_train[date.hour].append(consumption)

        dates += [date]
        consumptions += [consumption]
    print("From {0} to {1}".format(dates[0], dates[-1]))
    # now i create one svm for each hour
    train_and_predict(predicted)

#di nuovo calcolo ora per ora, alleno un modello per ogni ora?
def train_and_predict(predicted):
    for hour in X_train:
        model = MLPRegressor(hidden_layer_sizes=10,activation="relu", batch_size=12)
        model.fit(X_train[hour], Y_train[hour])

        predicted += [model.predict(X_test[hour])]
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

prepare_model_and_predict(values, dates, consumptions, predict_y_array)
accuracy = mean_accuracy_with_confidence_interval(predict_y_array, interval_size=0.3, Y_test=Y_test)
print("Mean accuracy: " + str(accuracy))

plt.plot(dates[-26:-2],consumptions[-26:-2], color='red')
plt.plot(dates[-26:-2], predict_y_array)
plt.xlabel('dates')
plt.ylabel('Kwh')
plt.gcf().autofmt_xdate()
plt.show()