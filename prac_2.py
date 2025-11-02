import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Sample dataset
data = {
    'filename': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'],
    'width': [640, 800, np.nan, 1024],
    'height': [480, 600, 720, np.nan],
    'category': ['cat', 'dog', 'cat', None]
}

df = pd.DataFrame(data)
print("Original Data:\n", df)

# Handle missing values
df['width'] = df['width'].fillna(df['width'].mean())
df['height'] = df['height'].fillna(df['height'].mean())
df['category'] = df['category'].fillna('unknown')

# Normalize numerical columns
scaler = MinMaxScaler()
df[['width', 'height']] = scaler.fit_transform(df[['width', 'height']])

# Encode categorical variable
encoder = LabelEncoder()
df['category_encoded'] = encoder.fit_transform(df['category'])

print("\nCleaned & Preprocessed Data:\n", df)
