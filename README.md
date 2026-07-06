# Data-preprocessing-
# Hotel Booking Cancellation Prediction 

A simple machine learning project that predicts whether a hotel booking will be **canceled or not**, based on details like lead time, number of guests, room type, deposit type, etc.

This was built as a data preprocessing + classification practice project, using a real-world style dataset (`hotel_booking.csv`).


##  What This Project Does

1. Loads and cleans hotel booking data (removes duplicates and duplicate/leaky columns).
2. Splits the data into numeric and categorical columns.
3. Builds a preprocessing pipeline to:
   - Fill missing numeric values with the **median**
   - Fill missing categorical values with the **most frequent value**
   - One-hot encode categorical columns
4. Trains a **Logistic Regression** model to predict the `is_canceled` column.
5. Evaluates the model using Accuracy, ROC-AUC score, Confusion Matrix, and a full Classification Report.

Along with this, there's a small bonus script (`booking2.py`) that visually demonstrates the **Curse of Dimensionality** — showing how distances between data points behave differently as the number of features increases (2D vs 10D vs 50D vs 200D).


##  About Data Leakage

While preprocessing, two columns — `reservation_status` and `reservation_status_date` — were **intentionally dropped**.

These columns basically reveal the outcome of the booking (cancelled or not) *after the fact*, so keeping them would let the model "cheat" by peeking at the answer. This is called **data leakage**, and removing it is an important step to make sure the model's evaluation reflects real-world performance.



##  Tech Stack / Libraries Used

- Python 
- Pandas & NumPy — data handling
- Scikit-learn — preprocessing pipeline & Logistic Regression model
- Matplotlib & SciPy — for the dimensionality visualization script



##  Project Files

| File | Description |
|------|-------------|
| `booking.py` | Main script: loads data, preprocesses it, trains a Logistic Regression model, and evaluates results |
| `booking2.py` | A small side experiment showing how distances between points spread out as dimensions increase |
| `hotel_booking.csv` | The dataset used (not included in repo — see below) |



## How to Run

1. Clone this repository
   ```bash
   git clone <your-repo-link>
   cd <your-repo-folder>
   ```

2. Install the required libraries
   ```bash
   pip install pandas numpy scikit-learn matplotlib scipy
   ```

3. Make sure `hotel_booking.csv` is in the same folder as the script (this dataset is publicly available on Kaggle — search "Hotel Booking Demand").

4. Run the main script
   ```bash
   python booking.py
   ```

5. (Optional) Run the dimensionality demo
   ```bash
   python booking2.py
   ```


##  Sample Output

The main script prints:
- Accuracy score
- ROC-AUC score
- Confusion Matrix
- Full classification report (precision, recall, f1-score)



##  What I Learned

- How to build a clean **preprocessing pipeline** using `ColumnTransformer` and `Pipeline` instead of manually transforming each column.
- Why checking for **data leakage** is a critical step before training any model.
- How to handle **missing values** differently for numeric vs categorical data.
- A visual intuition for the **curse of dimensionality** — as dimensions increase, distances between points tend to become more similar, which affects distance-based ML algorithms.






