import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Generate Dataset
# -----------------------------
np.random.seed(7)

NUM_STUDENTS = 500

df = pd.DataFrame({
    "StudentID": [f"S{i}" for i in range(1, NUM_STUDENTS + 1)],
    "Math": np.random.randint(35, 101, NUM_STUDENTS),
    "Physics": np.random.randint(30, 101, NUM_STUDENTS),
    "Chemistry": np.random.randint(40, 101, NUM_STUDENTS),
    "Attendance": np.random.randint(60, 101, NUM_STUDENTS)
})

# -----------------------------
# Feature Engineering
# -----------------------------
df["Total Marks"] = df[["Math", "Physics", "Chemistry"]].sum(axis=1)

df["Average Marks"] = (
    df["Total Marks"] / 3
).round(2)

# Grade Classification
conditions = [
    df["Average Marks"] >= 90,
    df["Average Marks"] >= 80,
    df["Average Marks"] >= 70,
    df["Average Marks"] >= 60
]

grades = ["A", "B", "C", "D"]

df["Grade"] = np.select(
    conditions,
    grades,
    default="F"
)

# Result Classification
df["Result"] = np.where(
    df["Average Marks"] >= 60,
    "Pass",
    "Fail"
)

# -----------------------------
# KPIs
# -----------------------------
print("\n===== KPI REPORT =====")

print("Total Students:", len(df))
print("Average Score:", round(df["Average Marks"].mean(), 2))
print("Highest Score:", round(df["Average Marks"].max(), 2))
print("Lowest Score:", round(df["Average Marks"].min(), 2))

print("\nGrade Distribution")
print(df["Grade"].value_counts())

print("\nTop 5 Students")
print(
    df.nlargest(
        5,
        "Average Marks"
    )[
        ["StudentID", "Average Marks"]
    ]
)

# -----------------------------
# Dashboard
# -----------------------------
plt.style.use("ggplot")

fig, ax = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)

fig.suptitle(
    "Student Performance Analytics Dashboard",
    fontsize=16,
    fontweight="bold"
)

# Top 20 Students
top20 = df.nlargest(
    20,
    "Average Marks"
)

ax[0].bar(
    top20["StudentID"],
    top20["Average Marks"]
)

ax[0].set_title("Top 20 Students")
ax[0].tick_params(rotation=90)

# Grade Distribution
grade_count = df["Grade"].value_counts()

ax[1].bar(
    grade_count.index,
    grade_count.values
)

ax[1].set_title("Grade Distribution")
ax[1].set_xlabel("Grade")
ax[1].set_ylabel("Count")

# Pass / Fail
result_count = df["Result"].value_counts()

ax[2].pie(
    result_count,
    labels=result_count.index,
    autopct="%1.1f%%",
    startangle=90
)

ax[2].set_title("Pass vs Fail")

plt.tight_layout()

plt.savefig(
    "Student_Performance_Dashboard.png"
)

plt.show()