import matplotlib.pyplot as plt

dog = [46/(46+48),  119/(119+131), 57/(57+78)]
activity = ["7.8.184.113",  "7.9.186.8155", "7.10.187.9675"]

fig, ax = plt.subplots()
ax.plot(activity, dog, label="")
ax.legend()
plt.xlabel('version', fontweight='bold')
plt.ylabel('win percentage', style='italic', loc='bottom')
plt.title('Win percentage of Char id 6: Urgot')
plt.show()






