import streamlit as st
import pandas as pd

def main():
    st.title("Equipment Failure Prediction")
    selected_page = st.page_link("app.py", label="Home", icon="🏠")
    selected_page = st.page_link('pages/dataVisualization.py', label="Data Visualization And Analysis", icon="📊")
    selected_page = st.page_link('pages/prediction.py', label="Prediction", icon="⚙")

    tab1, tab2 = st.tabs(["Introduction", "🗃 Data"])
    
    # Display content in Tab 1
    tab1.markdown("""
    ### Equipment Failure Prediction Using Machine Learning
In today's industrial landscape, the ability to predict equipment failures before they occur is crucial for maintaining operational efficiency, reducing downtime, and preventing costly repairs. Leveraging the power of machine learning, the Equipment Failure Prediction project aims to revolutionize the way industries manage their assets by forecasting potential failures based on various operational parameters.

Project Overview
The Equipment Failure Prediction project utilizes advanced machine learning algorithms to analyze sensor data and other relevant operational parameters collected from industrial equipment. By identifying patterns and trends in the data, the project aims to predict potential equipment failures before they occur, allowing proactive maintenance and minimizing unplanned downtime.

Key Features
Data Collection and Preprocessing: The project involves collecting sensor data and operational parameters from industrial equipment, cleaning and preprocessing the data to remove noise and inconsistencies, and preparing it for analysis.
Feature Engineering: Utilizing domain knowledge and advanced techniques, the project engineers informative features from the raw data to enhance the predictive power of the machine learning models.
Model Development: Various machine learning models, including Random Forest, LSTM (Long Short-Term Memory), and others, are developed and trained on historical data to predict equipment failures.
Model Evaluation and Validation: The performance of the trained models is evaluated using metrics such as accuracy, precision, recall, and F1-score. Cross-validation techniques are employed to ensure robustness and generalizability.
Deployment and Integration: The best-performing model is deployed into production, integrated with existing systems, and used to provide real-time predictions of equipment failures.
Benefits
Reduced Downtime: By predicting equipment failures in advance, industries can schedule maintenance activities during planned downtimes, minimizing disruption to operations.
Cost Savings: Proactive maintenance reduces the need for costly emergency repairs and extends the lifespan of equipment, resulting in significant cost savings.
Improved Safety: Predicting equipment failures enhances workplace safety by preventing accidents and incidents caused by malfunctioning machinery.
    """)



### ------------------------------------------------------------
# Tab 2

# Display content in Tab 2
    tab2.markdown("""
    ### Data Source

The data is sourced directly from sensors installed on various types of industrial equipment, including but not limited to manufacturing machinery, processing units, and infrastructure components. These sensors continuously monitor parameters such as temperature, pressure, rotational speed, torque, and tool wear, providing valuable insights into the health and performance of the equipment.

### Features

The dataset comprises a diverse set of features, each representing a specific operational parameter or sensor reading. Some of the key features include:

- **Air Temperature [K]**: The ambient air temperature surrounding the equipment, measured in Kelvin.
- **Process Temperature [K]**: The temperature of the manufacturing or processing process, measured in Kelvin.
- **Rotational Speed [rpm]**: The speed at which the equipment's rotating components are operating, measured in revolutions per minute.
- **Torque [Nm]**: The amount of rotational force applied to the equipment, measured in Newton-meters.
- **Tool Wear [min]**: The duration for which the equipment's cutting or machining tool has been in use, measured in minutes.

    """)
    # Load data
    df = pd.read_csv("SENSOR DATA/SensorData.csv")
    # Create an expander within Tab 2
    see_data=tab2.expander('You can click here to see the raw data first 👉')
    # With the expander context, display dataframe
    with see_data:
        st.dataframe(data=df.reset_index(drop=True))

if __name__ == "__main__":
    main()
