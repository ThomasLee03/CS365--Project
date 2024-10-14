
import numpy as np 
import matplotlib.pyplot as plt 
 
# set width of bar 
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8)) 
 
# set height of bar 
TRAcc = [0.9397007042253521, 1.0, 0.9036091549295775, 1.0] 
TAcc = [ 0.8867605633802817, 0.7887323943661971, 0.8626760563380281, .8767605633802817] 

#categorical NB

#Training accuracy:  0.9036091549295775
#Test accuracy:  0.8626760563380281

#random forest:
    
#Training accuracy:  1.0
#Test accuracy:  0.8767605633802817    


#decision tree:
    
#Training accuracy:  1.0
#Test accuracy:  0.7887323943661971

 
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
        ['LR: nc (OHE, l2)', 'DTC', 'Cat NB', 'Random Forest'])
 
 

plt.ylim([.51, 1.1])

plt.title('Training and Testing Accuracy for different models')
plt.legend()
plt.show() 
