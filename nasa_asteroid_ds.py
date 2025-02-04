'''
My name is: Sara Lasri (id: 326206984)
Nice check!!
'''
import numpy as np
import pandas as pd
from scipy.stats import linregress as lr
import matplotlib.pyplot as plt
import os

def load_data(file_name):
    # Check if the file name is valid
    if not isinstance(file_name, str) or not file_name.endswith('.csv'):
        print(f"Error: The file name: '{file_name}' is not valid. Please provide a valid CSV file.")
        return None
    # Check if the file exists
    if not os.path.exists(file_name):
        print(f"Error: The file '{file_name}' is not available.")
        return None
    try:
        # Try loading the CSV file into a DataFrame
        df = pd.read_csv(file_name)
        return df

    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_name}' is empty and does not contain any data.")
    # except pd.errors.ParserError:
    #     print(f"Error: The file '{file_name}' could not be parsed. It may have an invalid format.")
    except Exception as err:
        print(f"Error: An unexpected error occurred: {err}")

def mask_data(df):
    # Ensure the 'Date Approach Close' column is in datetime format
    df['Close Approach Date'] = pd.to_datetime(df['Close Approach Date'], errors='coerce')
    # Filter asteroids with 'Date Approach Close' from the year 2000 onward
    df_filtered = df[df['Close Approach Date'] >= pd.Timestamp('2000-01-01')]
    return df_filtered

def data_details(df):
    # Clean up the DataFrame by dropping the specified columns
    columns_to_drop = ['Neo Reference ID', 'Orbiting Body', 'Equinox']
    df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')  # errors='ignore' to avoid errors if columns don't exist
    # Create the tuple with 3 elements:
    num_rows = df_cleaned.shape[0]  # number of rows
    num_columns = df_cleaned.shape[1]  # number of columns
    column_headers = df_cleaned.columns.tolist()  # list of column headers
    # Return the tuple
    return (num_rows, num_columns, column_headers)

def max_absolute_magnitude(df):
    # Filter out rows with missing values in 'Absolute Magnitude' or 'Distance'
    df_cleaned = df.dropna(subset=['Absolute Magnitude'])
    # Find the row with the maximum 'Absolute Magnitude' (which means closest to Earth)
    max_magnitude_row = df_cleaned.loc[df_cleaned['Absolute Magnitude'].idxmax()]
    # Return the tuple with asteroid name and the maximum absolute magnitude
    return (max_magnitude_row['Name'].item(), max_magnitude_row['Absolute Magnitude'].item())

def closest_to_earth(df):
    # Filter out rows with missing values in 'Miss Dist.' (kilometers)
    df_cleaned = df.dropna(subset=['Miss Dist.(kilometers)'])
    # Find the row with the minimum 'Miss Dist.' (the closest asteroid to Earth)
    closest_asteroid_row = df_cleaned.loc[df_cleaned['Miss Dist.(kilometers)'].idxmin()]
    # Return the name of the closest asteroid
    return closest_asteroid_row['Name']

def common_orbit(df):
    # Count the number of asteroids for each 'ID Orbit'
    orbit_counts = df['Orbit ID'].value_counts().to_dict()
    # return orbit_counts

    # Sort the dictionary by value in descending order
    sorted_orbit_counts = dict(sorted(orbit_counts.items(), key=lambda item: item[1], reverse=True))
    return sorted_orbit_counts

def min_max_diameter(df):
    # Calculate the average of the 'Max Dia Est in KM' column
    avg_diameter = df['Est Dia in KM(max)'].mean()
    # Filter the DataFrame to find asteroids with diameter above the average
    above_avg_asteroids = df[df['Est Dia in KM(max)'] > avg_diameter]
    # Return the count of asteroids with diameter above average
    return above_avg_asteroids.shape[0]

def plt_hist_diameter(df):
    # Calculate the average diameter for each asteroid
    df['Average Diameter'] = df[['Est Dia in KM(min)', 'Est Dia in KM(max)']].mean(axis=1)
    # Create histogram of the average diameter
    plt.figure(figsize=(10, 6))
    plt.hist(df['Average Diameter'], bins=100, color='skyblue', edgecolor='black')
    # Set titles and labels for the graph
    plt.title('Distribution of Average diameter size', fontsize=14)
    plt.xlabel('Average Diameter (in KM)', fontsize=12)
    plt.ylabel('Number of Asteroids', fontsize=12)
    # Display the graph
    plt.grid(True)
    plt.show()

def plt_hist_common_orbit(df):
    # Ensure that 'Minimum Orbit Intersection' is in numeric format (if not already)
    df['Minimum Orbit Intersection'] = pd.to_numeric(df['Minimum Orbit Intersection'], errors='coerce')
    # Create histogram for 'Minimum Orbit Intersection' with 10 bins
    plt.figure(figsize=(10, 6))
    plt.hist(df['Minimum Orbit Intersection'], bins=10, color='lightgreen', edgecolor='black')
    # Set titles and labels for the graph
    plt.title('Distribution of Asteroids by Minimum Orbit Intersection', fontsize=14)
    plt.xlabel('Minimum Orbit Intersection', fontsize=12)
    plt.ylabel('Number of Asteroids', fontsize=12)
    # Display the graph
    plt.grid(True)
    plt.show()

def plt_pie_hazard(df):
    # Count the occurrences of each category (True for Hazardous, False for Non-Hazardous)
    hazard_counts = df['Hazardous'].value_counts()
    # Plot the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(hazard_counts, labels=hazard_counts.index.map({True: 'True', False: 'False'}), autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'],
            explode=(0.1, 0))
    # Set the title
    plt.title('Percentage of Hazardous and Non-Hazardous Asteroids', fontsize=14)
    # Display the pie chart
    plt.show()

def plt_linear_motion_magnitude(df):
    # Drop rows with missing values in either 'Miss Dist.(kilometers)' or 'Speed (hour per Miles)'
    df_cleaned = df.dropna(subset=['Miss Dist.(kilometers)', 'Miles per hour'])

    # Extract the two columns
    x = df_cleaned['Miss Dist.(kilometers)']
    y = df_cleaned['Miles per hour']

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = lr(x, y)

    # Create the regression line
    regression_line = slope * x + intercept

    # Plot the data and the regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Data points', color='blue', alpha=0.5)
    plt.plot(x, regression_line, color='red', label='Regression line')

    # Set the labels and title
    plt.xlabel('Absolute Magnitude')
    plt.ylabel('Miles per hour')
    plt.title('Linear Regression: Absolute Magnitude vs Miles per hour')
    plt.legend()
    # Show the plot
    plt.show()
    # Print the Pearson correlation coefficient
    print(f"Pearson correlation coefficient: {r_value:.2f}")
    # Return the correlation coefficient
    return r_value

df = load_data("nasa.csv")
df = mask_data(df)
c = data_details(df)
print(c)
d = max_absolute_magnitude(df)
print(d)
e = closest_to_earth(df)
print(e)
f = common_orbit(df)
print(f)
g = min_max_diameter(df)
print(g)
plt_hist_diameter(df)
plt_hist_common_orbit(df)
plt_pie_hazard(df)
plt_linear_motion_magnitude(df)
