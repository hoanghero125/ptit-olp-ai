import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

# Đọc dữ liệu
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Xử lý văn bản với TF-IDF
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(train['Review'])
X_test = vectorizer.transform(test['Review'])

# Mục tiêu cần dự đoán
y_train = train['Rating']

# Huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán trên tập test
predictions = model.predict(X_test)

# Tạo file submission
submission = pd.DataFrame({
    'Id': test['Id'],
    'Rating': predictions
})

# Lưu file submission
submission.to_csv('prediction.csv', index=False)