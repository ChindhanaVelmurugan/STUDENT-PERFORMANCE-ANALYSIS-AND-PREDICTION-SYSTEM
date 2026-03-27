# ===============================
# STUDENT PERFORMANCE ANALYSIS AND PREDICTION SYSTEM
# ===============================

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy import stats

# Step 2: Create Student Dataset
np.random.seed(0)
student_data = {
    "StudentID": range(1, 21),
    "Age": np.random.randint(15, 20, 20),
    "Math": np.random.randint(50, 100, 20),
    "Science": np.random.randint(45, 100, 20),
    "English": np.random.randint(40, 100, 20),
    "Attendance": np.random.randint(60, 100, 20),
    "Extracurricular": np.random.randint(0, 10, 20)
}
df = pd.DataFrame(student_data)
df["Performance"] = (df["Math"]*0.4 + df["Science"]*0.35 + df["English"]*0.25 + df["Extracurricular"]*1.5)

# Step 3: Handle Missing Values
df.loc[2, "Math"] = np.nan
df.loc[5, "Attendance"] = np.nan
df["Math"].fillna(df["Math"].mean(), inplace=True)
df["Attendance"].fillna(df["Attendance"].mean(), inplace=True)

# Step 4: Filter High Performers
high_performers = df[df["Performance"] > 85]

# Step 5: Random Sampling
sample_students = df.sample(5)

# Step 6: Hypothesis Testing
math_scores = df["Math"]
z_score = (np.mean(math_scores) - 75) / (np.std(math_scores)/np.sqrt(len(math_scores)))
print("Z-score for average Math score 75:", z_score)

# Step 7: Correlation & Heatmap
corr = df.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Step 8: Linear Regression - Predict Performance
X_lin = df[["Math", "Science"]]
y_lin = df["Performance"]
X_train, X_test, y_train, y_test = train_test_split(X_lin, y_lin, test_size=0.2, random_state=42)
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred = lin_model.predict(X_test)
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred))

# Step 9: Logistic Regression - Classify High/Low Performers
df["High_Performer"] = df["Performance"].apply(lambda x: 1 if x > 80 else 0)
X_log = df[["Math", "Science", "English"]]
y_log = df["High_Performer"]
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.2, random_state=42)
log_model = LogisticRegression()
log_model.fit(X_train_log, y_train_log)
y_pred_class = log_model.predict(X_test_log)
print("Logistic Regression Accuracy:", accuracy_score(y_test_log, y_pred_class))

# ===============================
# MULTIPLE VISUALIZATIONS
# ===============================

# 1. Bar Chart: Top 5 Students
top_students = df.sort_values(by="Performance", ascending=False).head(5)
plt.figure(figsize=(8,5))
plt.bar(top_students["StudentID"], top_students["Performance"], color='green')
plt.xlabel("Student ID")
plt.ylabel("Performance Score")
plt.title("Top 5 Student Performance")
plt.show()

# 2. Line Plot: Performance Trends
plt.figure(figsize=(10,5))
plt.plot(df["StudentID"], df["Math"], marker='o', label="Math")
plt.plot(df["StudentID"], df["Science"], marker='x', label="Science")
plt.plot(df["StudentID"], df["English"], marker='s', label="English")
plt.xlabel("Student ID")
plt.ylabel("Scores")
plt.title("Student Scores Trend")
plt.legend()
plt.show()

# 3. Scatter Plot: Math vs Science
plt.figure(figsize=(7,5))
plt.scatter(df["Math"], df["Science"], c=df["Performance"], cmap='viridis', s=100)
plt.colorbar(label="Performance")
plt.xlabel("Math Score")
plt.ylabel("Science Score")
plt.title("Math vs Science Scatter Plot")
plt.show()

# 4. Histogram: Performance Distribution
plt.figure(figsize=(8,5))
plt.hist(df["Performance"], bins=10, color='orange', edgecolor='black')
plt.xlabel("Performance Score")
plt.ylabel("Number of Students")
plt.title("Performance Score Distribution")
plt.show()

# 5. Box Plot: Scores by Subject
plt.figure(figsize=(8,5))
sns.boxplot(data=df[["Math", "Science", "English"]])
plt.title("Box Plot of Scores by Subject")
plt.show()

# 6. Pie Chart: High vs Low Performers
perform_counts = df["High_Performer"].value_counts()
plt.figure(figsize=(6,6))
plt.pie(perform_counts, labels=["Low Performer","High Performer"], autopct='%1.1f%%', colors=['red','green'])
plt.title("High vs Low Performers")
plt.show()

# 7. Heatmap: Correlation (Already included above, repeated for reference)
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of All Attributes")
plt.show()

# 8. Pair Plot: Relationship Between Scores
sns.pairplot(df[["Math","Science","English","Performance"]], hue="High_Performer", palette="Set2")
plt.suptitle("Pair Plot of Scores and Performance", y=1.02)
plt.show()
