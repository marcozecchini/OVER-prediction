import datetime as dt
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import statistics as st
import numpy as np

def mean_accuracy_with_confidence_interval(predict_array, interval_size, Y_test):
    accuracy = 0
    for hour in range(0,len(predict_array)):
        value_interval = predict_array[hour] * interval_size
        if (predict_array[hour] + value_interval >= Y_test[hour]) and (predict_array[hour] - value_interval <= Y_test[hour]):
            accuracy += 1
    return accuracy / 24

def root_mean_square_deviation(predict_y_array, Y_test):
    sum_squared = 0
    for hour in range(0, len(predict_y_array)):
        sum_squared += np.square(predict_y_array[hour] - Y_test[hour])
    return  np.sqrt(sum_squared/24)

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
                vect_temp += [consumption]
        x_vector += [st.median(vect_temp)]

        #addition of the temperature
        x_vector += [float(temp[2])] if len(temp)>2 else None

        day = dt.datetime.now().replace(hour=0,minute=0,second=0,microsecond=0).timestamp() - 24 * 3600 * days
        if (date.timestamp() >= day):
            if (date.timestamp() > day + 24 * 3600 - 1): continue
            X_test[date.hour].append(x_vector)
            Y_test.append(consumption)
            test_date.append(date)
        else:
            X_train[date.hour].append(x_vector)
            Y_train[date.hour].append(consumption)

        dates += [date]
        consumptions += [consumption]
    print("From {0} to {1}".format(dates[0], dates[-1]))
    train_and_predict(predicted)

#For each our I build a model and I test it
def train_and_predict(predicted):
    for hour in X_train:
        model = lm.LinearRegression()
        model.fit(X_train[hour], Y_train[hour])

        predicted += [model.predict(X_test[hour])]
        print("{0} value: {1}".format(hour, predicted[hour]))

################################################################################################################################################

#TO BE SET CORRECTLY
which_building = int(input("Which building do you want to model? Please just insert the number: ")) #WHICH BUILDING DO YOU WANT TO PLOT?
days = int(input("How many days ago do you want to predict? Plese, insert the number: ")) #TO CHOOSE THE DAY TO PREDICT
which_building2 = which_building
file = open("../real_consumption.txt")
while(which_building > 1):
    file.readline().strip()
    which_building -= 1
line = file.readline().strip()
building = line.split("\t")[0]
values = line.split("\t")[1].split(" ")[1:]

file2 = open("../temperature.txt")
while(which_building2 > 1):
    file2.readline().strip()
    which_building2 -= 1
line2 = file2.readline().strip()
building2 = line2.split("\t")[0]
temperatures = line2.split("\t")[1].split(" ")[1:]

if (len(values) == len(temperatures)):
    for i in range(0, len(values)):
        values[i] = values[i] + "," + temperatures[i].split(",")[1]

dates = [] #Stores all the dates
consumptions = [] #Stores all the consumptions
X_train = {} #dictionary used to keep track of the training set in hours
X_test = {} #dictionary used to keep track of the testing set in hours
Y_train = {} #dictionary used to keep track of the traget values of the training set in hours
Y_test = [] #list that stores the target values (consumptions) of the test set
test_date = [] # list that stores the dates of the testing set
predict_y_array = [] #list that stores the predicted consumptions

for i in range(0,24):
    X_train[i] = []
    X_test[i] = []
    Y_train[i] = []

prepare_model_and_predict(values, dates, consumptions, predict_y_array)
accuracy = mean_accuracy_with_confidence_interval(predict_y_array, interval_size=0.6, Y_test=Y_test)
RMSD = root_mean_square_deviation(predict_y_array, Y_test)
print("Mean accuracy with confidence interval: " + str(accuracy))
print("Root mean squared deviation: " + str(RMSD))

plt.plot(test_date, Y_test, color='red')
plt.plot(test_date, predict_y_array, color='blue')
plt.errorbar(test_date, predict_y_array, yerr=st.median(predict_y_array)[0] * 0.6, fmt='--o')
plt.xlabel('dates')
plt.ylabel('Kwh')
plt.gcf().autofmt_xdate()
plt.show()