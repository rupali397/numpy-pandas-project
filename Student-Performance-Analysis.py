import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Dataset
df = pd.read_csv("C:/Users/sahil/Desktop/Coding/cscorner/student_data.csv")
print("\n----DataSet Loaded----")
print(df.head())

# 2. Understand the data
print("\n-----Dataset shape----")
print("Rows: ",df.shape[0])
print("Columns: ",df.shape[1])
print("----Dataset Information-----")
df.info()

print("\n-----Statistical Summary-----")
print(df.describe())

# 3. Data Cleaning

print("\n-----Missing Values-----")
print(df.isnull().sum())
print("\n-----Duplicates------")
print("Duplicate rows:",df.duplicated().sum())

df=df.drop_duplicates()

# 4. NUMPY Analysis
cgpa=df["CGPA"].to_numpy()
coding = df["Coding_Score"].to_numpy()
attendance = df["Attendance"].to_numpy()
print("\n-----NUMPY Analysis-----")
print("Average CGPA:",np.mean(cgpa))
print("Median CGPA:",np.median(cgpa))
print("CGPA Standard Deviation:",np.std(cgpa))

print("Highest CGPA:",np.max(cgpa))
print("Lowest CGPA:",np.min(cgpa))
highest_index = np.argmax(cgpa)
print("Student with highest CGPA:",df.iloc[highest_index] ["Student_ID"])
print("Average Coding Score:",np.mean(coding))
print("Average Attendance:",np.mean(attendance))

print("90th Percentile of CGPA:",np.percentile(cgpa,90))

# 5. PANDAS Analysis
print("\n----Pandas Analysis----")
print("Average CGPA:",df["CGPA"].mean())
print("Highest CGPA:",df["CGPA"].max())
print("Lowest CGPA:",df["CGPA"].min())

# 6. Branch-Wise Analysis
print("\n----Branch-Wise Analysis----")
branch_cgpa = df.groupby("Branch")["CGPA"].mean()
print("\nAverage CGPA by Branch:")
print(branch_cgpa)
branch_coding = df.groupby("Branch")["Coding_Score"].mean()
print("Average Coding Score by Branch:")
print(branch_coding)

# 7.Placement Analysis
print("\n------Placement Analysis------")
placement_count=df["Placed"].value_counts()
print(placement_count)
total_students = len(df)
placed_students = (df["Placed"] == "Yes").sum()
placement_percentage = (placed_students/total_students)*100
print("\nTotal Students:",total_students)
print("\nPlaced Students:",placed_students)
print("Placement Percentage:",round(placement_percentage,2),"%")

# 8. Branch-Wise Placement
print("\n-----Branch-Wise Placement-----")
branch_placement = pd.crosstab(df["Branch"],df["Placed"])
print(branch_placement)

# 9. Internship VS Placement
print("\n-----Internship VS Placement-----")
internship_placement = pd.crosstab(df["Internship"],df["Placed"],normalize="index")*100
print(internship_placement)

# 10. Coding Score VS Placement
print("-----Coding Score VS Placement-----")
coding_placement = df.groupby("Placed")["Coding_Score"].mean()
print(coding_placement)

# 11. CGPA VS Placement
print("\n-----CGPA VS Placement-----")
cgpa_placement = df.groupby("Placed")["CGPA"].mean()
print(cgpa_placement)

# 12. Package Analysis
print("-----Package Analysis-----")
placed_df=df[df["Placed"] == "Yes"]
average_package = placed_df["Package_LPA"].mean()
highest_package = placed_df["Package_LPA"].max()
lowest_package = placed_df["Package_LPA"].min()
print("Average Package:",round(average_package,2),"LPA")
print("Highest Package:",highest_package,"LPA")
print("Lowest Package:",lowest_package,"LPA")

# 13. Highest Package Student
highest_package_index = df["Package_LPA"].idxmax()
print("\n-----Highest Package Student-----")
print(df.loc[highest_package_index])

# 14. Top 5 students
print("\n----- Top 5 Student By CGPA------")
top_students = df.sort_values(
    "CGPA",
    ascending=False
).head(5)
print(
    top_students[
     [
        "Student_ID",
        "Branch",
        "CGPA",
        "Placed",
        "Package_LPA"
     ]
    ]
)

# 15.High Coding Score Students
print("--------High Coding Score Students-------")
high_coding = df[df["Coding_Score"] >= 85]
print(
    high_coding[
        [
        "Student_ID",
        "Coding_Score",
        "CGPA",
        "Placed"
        ]
    ]
)


# 16. Correlation Analysis
print("---------Correlation Analysis-------")
numeric_columns = [
    "Attendance",
    "Assignment_Score",
    "Internal_Marks",
    "Coding_Score",
    "Aptitude_Score",
    "Communication_Score",
    "CGPA",
    "Package_LPA"
]
correlation = df[numeric_columns].corr()
print(correlation)

# 17. Visulatization - CGPA
plt.figure(figsize=(8,5))
plt.hist(df["CGPA"],bins=6)
plt.title("CGPA Distribution")
plt.xlabel("CGPA")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("cgpa_distribution.png")
plt.show()

# 18. Visualization - Branch CGPA
plt.figure(figsize=(8,5))
branch_cgpa.plot(kind="bar")
plt.title("Average CGPA by Branch")
plt.xlabel("Branch")
plt.ylabel("Average CGPA")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    "average_cgpa_by_branch.png"
)
plt.show()

# 19. Visualization - Coding VS CGPA
plt.figure(figsize=(8,5))
plt.scatter(
    df["Coding_Score"],
    df["CGPA"]
)
plt.title("Coding Score vs CGPA")
plt.xlabel("Coding Score")
plt.ylabel("CGPA")
plt.tight_layout()
plt.savefig(
    "coding_vs_cgpa.png"
)
plt.show()

# 20. Visualization - Placement 
plt.figure(figsize=(7,5))
df["Placed"].value_counts().plot(
    kind="bar"
)
plt.title("Placement Status")
plt.xlabel("Placement")
plt.ylabel("Number of Students")

plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    "placement_status.png"
)
plt.show()

# 21. FINAL INSIGHTS
print("-------FINAL INSIGHTS------")
print("Overall Placement Rate:",round(placement_percentage,2),"%")
print("Average CGPA:",round(df["CGPA"].mean(),2))
print("Average Coding Score:",round(df["Coding_Score"].mean(),2))
print("Average Package:",round(average_package,2),"LPA")
print("Highest Package:",highest_package,"LPA")

print("-------PROJECT COMPLETED-------")

