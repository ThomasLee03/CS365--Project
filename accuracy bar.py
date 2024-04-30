
import numpy as np 
import matplotlib.pyplot as plt 
 
# set width of bar 
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8)) 
 
# set height of bar 
TRAcc = [0.8309859154929577, 0.8912852112676056,0.8908450704225352,  0.9392605633802817 , 0.9194542253521126, 0.9397007042253521] 
TAcc = [0.7904929577464789, 0.8690845070422535, 0.8655633802816901, 0.8838028169014085, 0.8785211267605634, 0.8867605633802817] 

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
        ['LR LBFGS','LR nc', 'LR nc(l2)', 'LR nc(OHE)', 'LR PCA nc(OHE, l2)', 'LR nc(OHE, l2)'])
 
 

plt.ylim([.51, .95])

plt.title('Training and Testing Accuracy for Logistic Regression model Variations')
plt.legend()
plt.show() 