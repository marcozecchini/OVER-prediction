import datetime as dt
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import statistics as st

def mean_accuracy_with_confidence_interval(predict_array, interval_size, Y_test):
    accuracy = 0
    for hour in range(0,len(predict_array)):
        value_interval = predict_array[hour] * interval_size
        if (predict_array[hour] + value_interval >= Y_test[hour]) and (predict_array[hour] - value_interval <= Y_test[hour]):
            accuracy += 1
    return accuracy / 24

def prepare_model_and_predict(values, dates, consumptions, predicted):
    for element in values:
        x_vector = []
        temp = element.split(",")
        date = dt.datetime.fromtimestamp(int(temp[0]) / 1000)
        consumption = float(temp[1])
        #consumption of the same hour 24 hour before
        x_vector += [consumptions[-24] if len(consumptions) > 24 else 0]

        #median consumption of the same hour during the past week
        vect_temp = []
        for i in range(1,8):
            if len(consumptions) > 24*i:
                vect_temp += [consumptions[-24*i]]
            else:
                vect_temp += [0]
        x_vector += [st.median(vect_temp)]
        #TODO MANCA LA TEMPERATURA

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
    train_and_predict(predicted)

#di nuovo calcolo ora per ora, alleno un modello per ogni ora?
def train_and_predict(predicted):
    for hour in X_train:
        model = lm.LinearRegression()
        model.fit(X_train[hour], Y_train[hour])

        predicted += [model.predict(X_test[hour])]
        print(predicted[hour])


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