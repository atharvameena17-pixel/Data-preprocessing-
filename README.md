# Data-preprocessing-
Hotel Booking Cancellation Prediction 

A simple machine learning project that predicts whether a hotel booking will be canceled or not, based on details like lead time, number of guests, room type, deposit type, etc.

This was built as a data preprocessing + classification practice project, using a real-world style dataset (hotel_booking.csv).


.What This Project Does


Loads and cleans hotel booking data (removes duplicates and duplicate/leaky columns).
Splits the data into numeric and categorical columns.
Builds a preprocessing pipeline to:

Fill missing numeric values with the median
Fill missing categorical values with the most frequent value
One-hot encode categorical columns



Trains a Logistic Regression model to predict the is_canceled column.
Evaluates the model using Accuracy, ROC-AUC score, Confusion Matrix, and a full Classification Report.


Along with this, there's a small bonus script (booking2.py) that visually demonstrates the Curse of Dimensionality — showing how distances between data points behave differently as the number of features increases (2D vs 10D vs 50D vs 200D).


.About Data Leakage

While preprocessing, two columns — reservation_status and reservation_status_date — were intentionally dropped.

These columns basically reveal the outcome of the booking (cancelled or not) after the fact, so keeping them would let the model "cheat" by peeking at the answer. This is called data leakage, and removing it is an important step to make sure the model's evaluation reflects real-world performance.


.Tech Stack / Libraries Used


Python 
Pandas & NumPy — data handling
Scikit-learn — preprocessing pipeline & Logistic Regression model
Matplotlib & SciPy — for the dimensionality visualization script


