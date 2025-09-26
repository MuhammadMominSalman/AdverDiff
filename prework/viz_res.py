import pandas as pd
import matplotlib.pyplot as plt
# Load your data from the CSV file
df = pd.read_csv('img_class_train.csv')

# Count occurrences of each weather type
weather_counts = df['weather'].value_counts()

# Count occurrences of each time of day
time_of_day_counts = df['timeofday'].value_counts()

# Define IEEE-style fonts
label_font = {'fontsize': 8}
title_font = {'fontsize': 8, 'fontweight': 'bold'}
tick_fontsize = 8

# Set figure size (IEEE column width ~3.5 in)
plt.rcParams.update({
    'figure.figsize': (3.5, 5),
    'axes.linewidth': 0.5,
})
plt.rcParams["font.family"] = "Times New Roman"

# --- Plot Bar Graphs ---
plt.figure()

# Weather bar chart
plt.subplot(2, 1, 1)

plt.bar(weather_counts.index, weather_counts.values, width=0.4, color='skyblue')
plt.xlabel('Weather Condition', **label_font)
plt.ylabel('Number of Images', **label_font)
plt.title('Weather', **title_font)
plt.xticks(rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()

plt.subplot(2, 1, 2)
plt.bar(time_of_day_counts.index, time_of_day_counts.values, width=0.4,color='orange')
plt.xlabel('Time of Day', **label_font)
plt.ylabel('Number of Images', **label_font)
plt.title('Time of Day Distribution', **title_font)
plt.xticks(rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig('weather_distribution.svg')
plt.show()  # Display the plot