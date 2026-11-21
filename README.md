# car-evaluation-knn-classifier
Classifying car acceptability with K-Nearest Neighbors using Python, pandas and scikit-learn.

My first machine learning project. It predicts whether a car is unacceptable, acceptable, good or very good using the K-Nearest Neighbours (KNN) algorithm.

This is a supervised, multi-class classification problem. Each car is described by a few features (such as buying price, maintenance cost and safety), and the model predicts its quality class.
<img width="377" height="167" alt="image" src="https://github.com/user-attachments/assets/7b2eeb27-28ea-474d-9fc0-3b06c18c4821" />
<img width="374" height="177" alt="image" src="https://github.com/user-attachments/assets/21011b10-2d0d-4903-a67b-19d978513f21" />

Approach
1. Load the data with pandas (car.data).
2. Select features. Currently buying, maint and safety [update this if you use more].
3. Encode categories as numbers with a hand-made, ordered mapping:
   low = 0, med = 1, high = 2, vhigh = 3
4. This order matters for KNN because it measures distance. An alphabetical encoding (such as LabelEncoder) would make low   look bigger than high.
5. Split the data into 80% training and 20% testing (train_test_split, with random_state fixed so results are repeatable).
6. Train a KNeighborsClassifier.
7. Evaluate accuracy on the unseen test set.


What I learned
- Models work with numbers, so text categories have to be encoded, and the order of the numbers matters for distance-based models like KNN.
- Accuracy needs a baseline. 75% looked fine until I compared it with the 70% I would get from always guessing the most common class.
- random_state makes the train/test split repeatable, so results can be compared fairly.
- Debugging data types (object vs int) and NaN values from a mismatched mapping is a big part of the work.
