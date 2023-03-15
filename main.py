from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import csv

dataset = loadtxt('./pima-indians-diabetes.csv', delimiter=',')

X = dataset[:, 0:8]
y = dataset[:, 8]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# define our model
model = Sequential()
# add the first layer
model.add(Dense(12, input_shape=(8, ), activation='relu'))
# add the second layer
model.add(Dense(8, activation='relu'))
# add the output layer
model.add(Dense(1, activation='sigmoid'))

# compile the network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit our model
model.fit(X_train, y_train, epochs=150, batch_size=10)

# evaluate the model
placeholder, accuracy = model.evaluate(X, y)
print(f"Accuracy: {accuracy}")

# make predictions on test data
predictions = model.predict(X_test)

# save predictions
with open ('predictions.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'prediction'])
    for i in range(len(predictions)):
        writer.writerow([i+1, predictions[i][0]])

# round predictions
rounded = [round(x[0]) for x in predictions]

