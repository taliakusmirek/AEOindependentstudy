import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))  # run from anywhere
import csv

STU = "Prospective Student"
PAR = "Parent of Prospective Student"

rows = [
    # (Topic, Prompt, Tags, Regions, Language, Platforms, Personas, Analysis Types)
    ("BC academic reputation", "What is Boston College known for academically?", "informational", "United States", "en-US", "", STU, "Visibility"),
    ("BC admissions", "What is the acceptance rate at Boston College?", "informational", "United States", "en-US", "", f"{STU}, {PAR}", "Visibility"),
    ("BC computer science program", "What majors does Boston College offer in computer science?", "informational", "United States", "en-US", "", STU, "Visibility"),
    ("BC campus life", "What is campus life like at Boston College?", "informational", "United States", "en-US", "", f"{STU}, {PAR}", "Visibility"),
    ("BC cost of attendance", "How much does it cost to attend Boston College?", "informational", "United States", "en-US", "", f"{STU}, {PAR}", "Visibility"),
    ("BC history", "What is the history of Boston College?", "informational", "United States", "en-US", "", STU, "Visibility"),
    ("BC financial aid", "What financial aid does Boston College offer?", "informational", "United States", "en-US", "", f"{STU}, {PAR}", "Visibility"),
    ("BC business school", "What is the Carroll School of Management known for?", "informational", "United States", "en-US", "", STU, "Visibility"),
    ("BC student housing", "What housing options are available for BC freshmen?", "informational", "United States", "en-US", "", f"{STU}, {PAR}", "Visibility"),
    ("BC study abroad", "What study abroad programs does Boston College have?", "informational", "United States", "en-US", "", STU, "Visibility"),
    ("BC vs Boston University", "How does Boston College compare to Boston University?", "comparison", "United States", "en-US", "", f"{STU}, {PAR}", "Visibility"),
    ("BC vs Northeastern", "Is Boston College better than Northeastern for business?", "comparison", "United States", "en-US", "", STU, "Visibility"),
    ("BC vs Notre Dame", "Boston College vs. Notre Dame: which has a stronger Jesuit tradition?", "comparison", "United States", "en-US", "", f"{STU}, {PAR}", "Visibility"),
    ("BC vs UMass Amherst", "How does BC's computer science program compare to UMass Amherst's?", "comparison", "United States", "en-US", "", STU, "Visibility"),
    ("BC vs Holy Cross", "Boston College vs. Holy Cross: which is a better fit for a pre-law student?", "comparison", "United States", "en-US", "", STU, "Visibility"),
    ("BC cost vs Catholic universities", "How does the cost of Boston College compare to other private Catholic universities?", "comparison", "United States", "en-US", "", f"{STU}, {PAR}", "Visibility"),
    ("BC vs Villanova", "BC vs. Villanova: which has better career outcomes in finance?", "comparison", "United States", "en-US", "", STU, "Visibility"),
    ("BC vs Boston University", "How does Boston College's campus compare to Boston University's in terms of location?", "comparison", "United States", "en-US", "", f"{STU}, {PAR}", "Visibility"),
    ("Jesuit/Catholic higher education tradition", "What are the best Jesuit colleges for combining academics with community service?", "recommendation", "United States", "en-US", "", STU, "Visibility"),
    ("Jesuit/Catholic higher education tradition", "Where can I find a Catholic college with an active faith community?", "recommendation", "United States", "en-US", "", f"{STU}, {PAR}", "Visibility"),
    ("Jesuit/Catholic higher education tradition", "Who offers the strongest financial aid for a Catholic college education?", "recommendation", "United States", "en-US", "", f"{STU}, {PAR}", "Visibility"),
    ("Rigorous academics", "What are the best colleges for a challenging liberal arts education?", "recommendation", "United States", "en-US", "", STU, "Visibility"),
    ("Rigorous academics", "Who offers the best undergraduate business education with challenging coursework?", "recommendation", "United States", "en-US", "", STU, "Visibility"),
    ("Holistic education (intellectual, ethical, and spiritual development)", "Where can I find a college centered on service to others?", "recommendation", "United States", "en-US", "", f"{STU}, {PAR}", "Visibility"),
    ("Holistic education (intellectual, ethical, and spiritual development)", "What are the best colleges for discussing life's big questions?", "recommendation", "United States", "en-US", "", STU, "Visibility"),
    ("Undergraduate education", "What are the best colleges for studying business as an undergraduate?", "recommendation", "United States", "en-US", "", STU, "Visibility"),
]

header = ["Topic", "Prompt", "Tags", "Regions", "Language", "Platforms", "Personas", "Analysis Types"]

if __name__ == "__main__":
    with open("prompts.csv", "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

    print(f"wrote {len(rows)} rows")
