import random
import tensorboard
import numpy as np
import matplotlib.pyplot as plt

classes = [str(i) for i in range(1, 200)]
n_classes = len(classes)

no_tpt_acc = {c: random.random() for c in classes}
tpt_acc = {c: random.random()* random.randint(-1, 1) + no_tpt_acc[c]  for c in classes}
top_val = max(max(no_tpt_acc.values()), max(tpt_acc.values()))

for key in no_tpt_acc:
    no_tpt_acc[key] = max(no_tpt_acc[key] / top_val, 0)
    
for key in tpt_acc:
    tpt_acc[key] = max(tpt_acc[key] / top_val, 0)


# plot a tensorboard histogram for each class where x-axis is the class name and y-axis is the accuracy.
# no_tpt_acc and tpt_acc have the same classes, so both class 'A's should be on the same x-axis but of different colors.
# generate the code using tensorboard if possible
fig, ax = plt.subplots(dpi=500)
ax.bar(no_tpt_acc.keys(), no_tpt_acc.values(), color='b', alpha=0.5, label='No TPT')
ax.bar(tpt_acc.keys(), tpt_acc.values(), color='r', alpha=0.5, label='TPT')

ax.set_ylabel('Accuracy')
ax.set_title('Accuracy by class')
ax.set_xticks(np.arange(n_classes))
ax.set_xticklabels(classes, rotation=-90, fontsize=2)
ax.legend()

fig.savefig('results/imagenet_A/plots/accuracy_by_class.png')

# how to include this in tensorboard
# tensorboard --logdir=runs

