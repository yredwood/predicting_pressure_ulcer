import numpy as np

input_data = 'input_datasets/LAB_CHART_EVENTS_TABLE.csv'
output_data = 'input_datasets/LAB_CHART_EVENTS_TABLE_sampled.csv'

ratio = 0.2

with open(input_data, 'r') as f:
    lines = f.readlines()

#sampling = np.random.choice(len(lines), size=int(len(lines)*ratio), replace=False)

np.random.seed(0)
out_lines = [lines[0].strip()]
for i, line in enumerate(lines[1:]):
    if np.random.rand() < ratio:
        out_lines.append(line.strip())


with open(output_data, 'w') as f:
    f.writelines('\n'.join(out_lines))
    
