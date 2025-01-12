import matplotlib.pyplot as plt
import numpy as np

# Data from the table
categories = ['Preference', 'Few Shot', 'Ours', "Supervised"]
few_shot_values = [1, 2, 5, 10, 20, 50, 100]
data = {
    "Preference": {
        "safety": [-0.60, 2.00, -5.20, -1.40, -2.00, 1.60, -5.20],
        "utility": [7.40, 8.60, 9.60, 6.80, 8.60, 8.60, 8.00]
    },
    "Supervised": {
        "safety": [2.40, -3.20, -5.00, 0.80, -2.60, -5.40, 0.00],
        "utility": [6.60, 5.00, 4.20, 5.20, 11.40, 7.00, 6.00]
    },
    "Few Shot": {
        "safety": [-7.40, 4.00, -17.00, -11.20, 0, 0, 0],
        "utility": [-2.40, -13.80, -2.60, -10.60, 0, 0, 0]
    },
    "Ours": {
        "safety": [-12.60, -6.20, -15.20, -2.80, -3.60, 1.80, -2.20],
        "utility": [23.60, 21.60, 13.20, 27.80, 23.40, 9.00, 11.40]
    }
}

# Define colors for each category
colors = {
    "Preference": "#66CCFF",
    "Supervised": "#FC00A3",
    "Few Shot": "#75CAC3",
    "Ours": "#F34573"
}

# Plot Safety Scores
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(few_shot_values))

# for i, category in enumerate(categories):
#     ax.bar(
#         x + i * 0.2,
#         data[category]["safety"],
#         width=0.2,
#         label=category,
#         color=colors[category]
#     )

# # ax.set_title("Safety Win Rate by Few-Shot Count", fontsize=14)
# ax.set_xlabel("Few-Shot Count", fontsize=12)
# ax.set_ylabel("Win Rate (%)", fontsize=12)
# ax.set_xticks(x + 0.2)
# ax.set_xticklabels(few_shot_values)
# ax.legend()
# plt.show()
# plt.savefig("outputs/safety.png")
# plt.savefig("outputs/safety.pdf")


# # Plot Utility Scores
# fig, ax = plt.subplots(figsize=(12, 6))

# for i, category in enumerate(categories):
#     ax.bar(
#         x + i * 0.2,
#         data[category]["utility"],
#         width=0.2,
#         label=category,
#         color=colors[category]
#     )

# # ax.set_title("Utility Win Rate by Few-Shot Count", fontsize=14)
# ax.set_xlabel("Few-Shot Count", fontsize=12)
# ax.set_ylabel("Win Rate (%)", fontsize=12)
# ax.set_xticks(x + 0.2)
# ax.set_xticklabels(few_shot_values)
# ax.legend()
# plt.show()
# plt.savefig("outputs/utility.png")
# plt.savefig("outputs/utility.pdf")


for category in categories:
    ax.plot(
        few_shot_values,
        data[category]["safety"],
        marker='o',
        label=category,
        color=colors[category]
    )

ax.set_xlabel("Few-Shot Count", fontsize=12)
ax.set_ylabel("Safety Win Rate (%)", fontsize=12)
ax.set_title("Safety Win Rate by Few-Shot Count", fontsize=14)
ax.legend()
plt.grid(True)
plt.savefig("outputs/safety_lineplot.png")
plt.savefig("outputs/safety_lineplot.pdf")
plt.show()

# Re-plot and save the Utility Scores line plot
fig, ax = plt.subplots(figsize=(12, 6))

for category in categories:
    ax.plot(
        few_shot_values,
        data[category]["utility"],
        marker='o',
        label=category,
        color=colors[category]
    )

ax.set_xlabel("Few-Shot Count", fontsize=12)
ax.set_ylabel("Utility Win Rate (%)", fontsize=12)
ax.set_title("Utility Win Rate by Few-Shot Count", fontsize=14)
ax.legend()
plt.grid(True)
plt.savefig("outputs/utility_lineplot.png")
plt.savefig("outputs/utility_lineplot.pdf")
plt.show()