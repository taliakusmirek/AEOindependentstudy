import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))  # run from anywhere
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.set_xlim(0, 11)
ax.set_ylim(0, 4.2)
ax.axis('off')

stages = [
    ("Profound Export", "Raw CSV pull from personal\nProfound seat: domain, source_url,\nprompt, engine, per BC prompt"),
    ("Citation\nExtractor", "Parses raw export, dedupes\nrows, normalizes domains\nand engine labels"),
    ("Classifier", "Assigns each row to one\nof 6 taxonomy categories\n(rule-based / learned)"),
    ("Tagging\nLayer", "Appends intent, cohort,\nplatform columns for\ntime-series comparison"),
    ("Report\nGenerator", "Computes citation-share\ndiffs, flags moves, writes\nweekly brief.md"),
]

box_w, box_h = 1.9, 2.1
gap = 0.35
total_w = len(stages) * box_w + (len(stages) - 1) * gap
start_x = (11 - total_w) / 2
y = 1.1

centers = []
for i, (title, desc) in enumerate(stages):
    x = start_x + i * (box_w + gap)
    centers.append(x + box_w / 2)
    box = FancyBboxPatch((x, y), box_w, box_h,
                          boxstyle="round,pad=0.05,rounding_size=0.08",
                          linewidth=1.5, edgecolor='black', facecolor='white')
    ax.add_patch(box)
    ax.text(x + box_w / 2, y + box_h - 0.35, title, ha='center', va='top',
             fontsize=10.5, fontweight='bold', color='black')
    ax.text(x + box_w / 2, y + box_h - 0.75, desc, ha='center', va='top',
             fontsize=7.3, color='black', wrap=True)
    ax.text(x + box_w / 2, y - 0.25, f"Stage {i+1}", ha='center', va='top',
             fontsize=8, color='black', style='italic')

for i in range(len(stages) - 1):
    arrow = FancyArrowPatch((centers[i] + box_w/2 - 0.05, y + box_h/2),
                             (centers[i+1] - box_w/2 + 0.05, y + box_h/2),
                             arrowstyle='-|>', mutation_scale=18, linewidth=1.8, color='black')
    ax.add_patch(arrow)

ax.text(5.5, 3.85, "AEO Citation Pipeline: 5-Stage Architecture", ha='center', va='top',
         fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('architecture_diagram.png', dpi=200, bbox_inches='tight')
print("saved")
