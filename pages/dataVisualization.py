import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly_express as px
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the data
@st.cache_data
def load_data():
    file_path = 'SENSOR DATA/SensorData.csv'
    return pd.read_csv(file_path)

# Data visualization Page
def data_visualization():
    st.title('Data Visualization')
   
    # Load the data
    df = load_data()
    df['Type'] = df['Type'].astype(str)
    tabs = st.tabs(["Dataset","Histograms", "Missing Values",  "Countplot by Type", "Process Temperature",])

    with tabs[0]:
        st.header("Dataset")
        st.text("this is our dataset")
        st.write(df.head(10))
    
    with tabs[1]:
        st.subheader('Histograms')
        selected_feature_hist = st.selectbox('Select a feature for histogram', df.columns[2:])
        st.pyplot(plot_histograms(df[selected_feature_hist]))

    with tabs[2]:
        st.subheader('Missing Values')
        st.write(missing_data(df))

    with tabs[3]:
        st.subheader('Countplot by Type')
        type_count(df)

    with tabs[4]:
        st.subheader('Process Temperature for different Equipment')
        selected_type = st.selectbox('Select an equipment type', df['Type'].unique())
        plot_air_temp_for_each_type(df[df['Type'] == selected_type], 'Type', 'Process temperature [K]')


# Data Analysis Page

def data_analysis():
    st.title('Data Analysis')

    df = load_data()
    df['Air temperature [K]'] = df['Air temperature [K]'].astype(int)
    df['Process temperature [K]'] = df['Process temperature [K]'].astype(int)
    df['Torque [Nm]'] = df['Torque [Nm]'].astype(int)
    df_types = pd.get_dummies(df, columns=['Type'])

    tabs = st.tabs(["Total Failure Type Count","Failure Type %","Corelation", "Pairplot"])
    
    with tabs[1]:
        st.title('Machine Failure Count and Percentage')

        # Calculate machine failure count and percentage
        machine_failure_percentage = count_and_percentage_machine_failures(df_types)

        # Plot pie and doughnut charts for each type of machine failure
        for index, row in machine_failure_percentage.iterrows():
            st.subheader(f'Machine Failure Percentage for {index}')
            fig = px.pie(row, values=row.values, names=row.index, hole=0.4, title=f'{index} Machine Failure Percentage')
            st.plotly_chart(fig)
    
    with tabs[0]:
        st.subheader("Machine Failures Type Count:")
        st.bar_chart(count_machine_failures(df_types))

    with tabs[2]:
        st.subheader('Correlation Heatmap')
        st.pyplot(plot_correlation_heatmap(df))
    
    with tabs[3]:
        st.subheader('Pairplot for Selected Features')
        selected_type = st.selectbox('Select equipment type', df['Type'].unique())
        selected_features = st.multiselect('Select features for pairplot', df.columns[3:])
        filtered_df = df[df['Type'] == selected_type]

    # Generate pair plot for selected features
        pair_plots = pair_plot_for_selected_features(filtered_df[selected_features], selected_features)
        for img_bytes in pair_plots:
            st.image(img_bytes)


# Helper functions for visualizations

def plot_histograms(df):
    plt.figure(figsize=(10, 6))
    df.hist()
    plt.tight_layout()
    return plt
    
# Function to calculate missing data
def missing_data(df):
    missing_values = df.isnull().sum()
    percent_missing = (missing_values / len(df)) * 100
    missing_data = pd.concat([missing_values, percent_missing], axis=1, keys=['Total', 'Percent'])
    return missing_data

# Function to plot machine failure count by type
def type_count(df):
    # Calculate counts for each type
    type_counts = df['Type'].value_counts()

    # Convert to DataFrame
    type_counts_df = pd.DataFrame({'Type': type_counts.index, 'Count': type_counts.values})

    # Plot the bar chart
    st.bar_chart(type_counts_df, x='Type', y='Count')

# Function to plot line chart for a specific feature by equipment type
def plot_air_temp_for_each_type(df, type_column, feature_column):
    types = df[type_column].unique()
    for equipment_type in types:
        subset = df[df[type_column] == equipment_type]
        plt.figure(figsize=(10, 6))
        plt.plot(subset[feature_column])
        plt.title(f'{feature_column} for Equipment Type {equipment_type}')
        plt.xlabel('Index')
        plt.ylabel(feature_column)
        st.pyplot()




# Helper function for analysis 

def count_and_percentage_machine_failures(df):
    machine_failures = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    types = ['Type_H', 'Type_L', 'Type_M']

    # Initialize a dictionary to store the results
    results = {}

    # For each type, count the number of machine failures
    for t in types:
        df_type = df[df[t] == 1]
        results[t] = df_type[machine_failures].sum()

    # Convert the results to a DataFrame
    df_results = pd.DataFrame(results)

    # Transpose the DataFrame so that types are rows and machine failures are columns
    df_results = df_results.T

    # Calculate the percentage for each type of machine failure
    df_results_percentage = df_results.div(df_results.sum(axis=1), axis=0) * 100

    return df_results_percentage

    
def count_machine_failures(df):
    # List of machine failure types
    machine_failures = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

    # Count the number of machine failures
    machine_failures_count = df[machine_failures].sum()

    # Create a DataFrame for machine failures count
    machine_failures_df = pd.DataFrame(machine_failures_count, columns=['Count'])

    return machine_failures_df

# Function to plot correlation heatmap
def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    numeric_df = df.iloc[:, 3:]
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f")
    return plt

# Function to create pair plots for selected features
def pair_plot_for_selected_features(df, selected_features):
    pair_plots = []
    for i in range(len(selected_features)):
        for j in range(i+1, len(selected_features)):
            # Create pair plot
            pairplot = sns.pairplot(df[selected_features])
            # Convert plot to image
            img_bytes = BytesIO()
            pairplot.savefig(img_bytes, format='png')
            img_bytes.seek(0)
            pair_plots.append(img_bytes)
    return pair_plots



def main():
    st.title('Equipment Failure Analysis')
    selected_page = st.page_link("app.py", label="Home", icon="🏠")
    selected_page = st.page_link('pages/dataVisualization.py', label="Data Visualization And Analysis", icon="📊")
    selected_page = st.page_link('pages/prediction.py', label="Prediction", icon="⚙")
    

    st.sidebar.title('Options')
    page = st.sidebar.radio("Choose a page", ["Data Visualization", "Data Analysis"])

    if page == "Data Visualization":
        data_visualization()
    elif page == "Data Analysis":
        data_analysis()

if __name__ == "__main__":
    main()

