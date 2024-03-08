
import numpy as np 
import matplotlib.pyplot as plt 
 
# set width of bar 
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8)) 
 
# set height of bar 
TRAcc = [0.5697321216515207, 0.52738181974, 0.5696709953] 
TAcc = [ 0.5297, 0.51126, 0.529086666666] 

 
# Set position of bar on X axis 
br1 = np.arange(len(TRAcc)) 
br2 = [x + barWidth for x in br1]  
 
# Make the plot
plt.bar(br1, TRAcc, color ='r', width = barWidth, 
        edgecolor ='grey', label ='Training Acc') 
plt.bar(br2, TAcc, color ='g', width = barWidth, 
        edgecolor ='grey', label ='Testing Acc') 

 
# Adding Xticks 
plt.xlabel('Model Used', fontweight ='bold', fontsize = 15) 
plt.ylabel('Accuracy', fontweight ='bold', fontsize = 15) 
plt.xticks([r + barWidth-.125 for r in range(len(TRAcc))], 
        ['LR: nc (OHE, l2)', 'DTC', 'Cat NB'])
 
 

plt.ylim([.51, .58])

plt.title('Training and Testing Accuracy for different models')
plt.legend()
plt.show() 