import matplotlib.pyplot as plt

dog = [(398+374)/38426,  (890+824)/70514, (372+366)/45580]
activity = ["7.8.184.113",  "7.9.186.8155", "7.10.187.9675"]

fig, ax = plt.subplots()
ax.plot(activity, dog, label="")
ax.legend()
plt.xlabel('version', fontweight='bold')
plt.ylabel('Pick Rate', style='italic', loc='bottom')
plt.title('Pick Rate of Char id 420: Illaoi')
plt.show()






