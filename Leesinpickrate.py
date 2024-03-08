import matplotlib.pyplot as plt

dog = [(3039+3342)/38426, (8157+8375)/70514, (3540+3919)/45580]
activity = ["7.8.184.113", "7.9.186.8155", "7.10.187.9675"]

fig, ax = plt.subplots()
ax.plot(activity, dog, label="")
ax.legend()
plt.xlabel('version', fontweight='bold')
plt.ylabel('Pick Rate', style='italic', loc='bottom')
plt.title('Pick Rate of Char id 64: LeeSin')
plt.show()






