# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Data Loading (replace with your dataset)
data = {
    'Crop': ['Wheat', 'Wheat', 'Corn', 'Corn', 'Rice', 'Rice', 'Wheat', 'Corn', 'Rice', 'Corn'],
    'Stage': ['Germination', 'Fruiting', 'Germination', 'Fruiting', 'Fruiting', 'Harvesting', 'Fruiting', 'Harvesting', 'Germination', 'Harvesting'],
    'Method': ['Pesticide_A', 'Pesticide_B', 'Pesticide_C', 'Pesticide_D', 'Pesticide_E', 'Pesticide_F', 'Pesticide_A', 'Pesticide_C', 'Pesticide_E', 'Pesticide_D']
}

df = pd.DataFrame(data)

# Data Preprocessing
df_encoded = pd.get_dummies(df, columns=['Crop', 'Stage'])

# Train-Test Split
X = df_encoded.drop('Method', axis=1)
y = df_encoded['Method']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Model Prediction
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')



# Classification Report
print('Classification Report:')
print(classification_report(y_test, y_pred, zero_division=1))
# User Recommendation
def recommend_method(crop, stage):
    user_input = pd.DataFrame({
        'Crop': [crop],
        'Stage': [stage]
    })

    # One-hot encode user input and align columns
    user_input_encoded = pd.get_dummies(user_input, columns=['Crop', 'Stage'])

    # Ensure the order of columns is the same as in the training data
    user_input_encoded = user_input_encoded.reindex(columns=X.columns, fill_value=0)

    # Predict recommended method
    recommendation = model.predict(user_input_encoded)

    return recommendation[0]


#list of crops and stages
print("  CROP       STAGE")
print("1.Wheat   1.Germination")
print("2.Corn    2.Fruiting")
print("3.Rice    3.Harvesting")

user_crop = str(input("Enter crop name: "))
user_stage = str(input("Enter the crop stage: "))

recommended_method = recommend_method(user_crop, f'Stage_{user_stage}')
print(f'Recommended Protection Method: {recommended_method}')