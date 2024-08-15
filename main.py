import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Load the cereal dataset
data = pd.read_csv('data\\cereal.csv')
print(data.head())

# Select relevant features and target
X = data[['calories', 'protein', 'fat', 'fiber', 'vitamins']]
y = data['name']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the nearest neighbors model
nn_model = NearestNeighbors(n_neighbors=1)
nn_model.fit(X_scaled)

# Example: Desired nutritional values
input_values = {
   'calories': 50,
   'protein': 30,
   'fat': 0,
   'fiber': 15,
   'vitamins': 20
}

# Convert input values to DataFrame and scale them
input_df = pd.DataFrame([input_values])
input_scaled = scaler.transform(input_df)

# Find the nearest neighbor
distance, index = nn_model.kneighbors(input_scaled)

# Get the name of the closest cereal
closest_cereal = y.iloc[index[0][0]]
print(f"The closest cereal is: {closest_cereal}")

def predict_diet(calorier,protein,fat,fiber,vitamins):
    req=[calorier,protein,fat,fiber,vitamins]
