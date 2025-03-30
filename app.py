# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd

# # Load the trained model
# with open("house_price_model.pkl", "rb") as model_file:
#     model = pickle.load(model_file)

# # Load feature names
# feature_names = pd.read_csv("./data/train.csv").columns.tolist()

# st.title("🏡 House Price Prediction App")

# # Option to choose input method
# input_method = st.radio("Choose Input Method:", ["Manual Input", "Upload CSV"])

# if input_method == "Manual Input":
#     user_input = {}
#     for feature in feature_names:
#         user_input[feature] = st.number_input(f"Enter {feature}:", value=0.0)

#     if st.button("Predict Price"):
#         input_features = np.array([list(user_input.values())])
#         predicted_price = model.predict(input_features)[0]
#         st.success(f"🏠 Estimated House Price: ${predicted_price:,.2f}")

# elif input_method == "Upload CSV":
#     uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)
#         flag = True
#         for feat in feature_names:
#             if feat not in list(df.columns):
#                 flag = False
#                 break

#         if flag:
#             predictions = model.predict(df)
#             df["Predicted Price"] = predictions
#             st.write(df)
#             st.download_button("Download Predictions", df.to_csv(index=False), file_name="predictions.csv")
#         else:
#             st.error("CSV columns do not match expected features. Please upload a valid file.")


# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd

# # Load the trained model
# with open("house_price_model.pkl", "rb") as model_file:
#     model_data = pickle.load(model_file)

# model = model_data['model']

# # Extract feature names from the trained model
# feature_names = model_data["features"]

# st.title("🏡 House Price Prediction App")

# # Option to choose input method
# input_method = st.radio("Choose Input Method:", ["Manual Input", "Upload CSV"])

# if input_method == "Manual Input":
#     user_input = {}
#     for feature in feature_names:
#         user_input[feature] = st.number_input(f"Enter {feature}:", value=0.0)

#     if st.button("Predict Price"):
#         input_features = np.array([list(user_input.values())])
#         predicted_price = model.predict(input_features)[0]
#         st.success(f"🏠 Estimated House Price: ${predicted_price:,.2f}")

# elif input_method == "Upload CSV":
#     uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)

#         # Debugging Logs
#         st.write("📌 Uploaded CSV Preview:", df.head())
#         st.write("✅ Expected Columns:", feature_names)
#         st.write("📌 Actual Columns:", list(df.columns))
#         st.write("⚠️ Missing Columns:", set(feature_names) - set(df.columns))
#         st.write("⚠️ Extra Columns:", set(df.columns) - set(feature_names))

#         if set(feature_names).issubset(df.columns):  # Ensure all required columns are present
#             df = df[feature_names]  # Select only relevant columns
#             predictions = model.predict(df.values)  # Convert DataFrame to NumPy before prediction
#             df["Predicted Price"] = predictions
#             st.write(df)
#             st.download_button("Download Predictions", df.to_csv(index=False), file_name="predictions.csv")
#         else:
#             st.error("CSV columns do not match expected features. Please upload a valid file.")


import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model and feature names
with open("house_price_model.pkl", "rb") as model_file:
    model_data = pickle.load(model_file)

model = model_data['model']  # Extract model
feature_names = model_data['features']  # Extract feature names

# If your model has categorical features, specify them here
categorical_features = ['Neighborhood', 'House Style']  # Modify as per dataset

# Define category options (modify as per dataset)
category_options = {
    'Neighborhood': ['A', 'B', 'C', 'D'],
    'House Style': ['1-Story', '2-Story', 'Split-Level']
}

st.title("🏡 House Price Prediction App")


# Option to choose input method
input_method = st.radio("Choose Input Method:", ["Manual Input", "Upload CSV"])

if input_method == "Manual Input":
    user_input = {}
    for feature in feature_names:
        if feature in categorical_features:
            user_input[feature] = st.selectbox(f"Select {feature}:", category_options[feature])
        else:
            user_input[feature] = st.number_input(f"Enter {feature}:", value=0.0)

    if st.button("Predict Price"):
        # Convert categorical features to numerical encoding (modify based on actual preprocessing)
        for feature in categorical_features:
            user_input[feature] = category_options[feature].index(user_input[feature])
        
        input_features = np.array([list(user_input.values())])
        predicted_price = model.predict(input_features)[0]
        st.success(f"🏠 Estimated House Price: ${predicted_price:,.2f}")

elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Debugging logs
        st.write("📌 Uploaded CSV Preview:", df.head())
        st.write("✅ Expected Columns:", feature_names)
        st.write("📌 Actual Columns:", list(df.columns))
        st.write("⚠️ Missing Columns:", set(feature_names) - set(df.columns))
        st.write("⚠️ Extra Columns:", set(df.columns) - set(feature_names))


        if set(feature_names).issubset(df.columns):  # Ensure all required columns are present
            df = df[feature_names]  # Select only relevant columns
            
            # Convert categorical variables (modify if needed)
            for feature in categorical_features:
                df[feature] = df[feature].map(lambda x: category_options[feature].index(x) if x in category_options[feature] else np.nan)
                if not df[feature].mode().empty:
                    df[feature].fillna(df[feature].mode()[0], inplace=True)   # Handle missing categorical values
            
            df = df.apply(pd.to_numeric, errors='coerce')  # Ensure numerical data type
            predictions = model.predict(df.values)  # Convert DataFrame to NumPy before prediction
            df["Predicted Price"] = predictions
            st.write(df)
            st.download_button("Download Predictions", df.to_csv(index=False), file_name="predictions.csv")
        else:
            st.error("CSV columns do not match expected features. Please upload a valid file.")