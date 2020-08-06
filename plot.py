import matplotlib.pyplot as plt
import numpy as np

S2S = [[0, 0.06], [5, 0.51], [10, 0.57], [15, 0.68], [20, 0.67] ]
S2SA = [[0,0.44], [5,0.85], [10,1.57], [15,1.95], [20,2.75]]
HRED = [[0,2.45], [5,5.15], [10,5.62], [15,5.95], [20,6.13]]

# plot the index for the x-values
#plt.plot(xi, y, marker='o', linestyle='--', color='r', label='Square')

plt.plot([i[0] for i in S2S], [i[1] for i in S2S], marker='o', color='r')
plt.plot([i[0] for i in S2SA], [i[1] for i in S2SA], marker='x', color='b')
plt.plot([i[0] for i in HRED], [i[1] for i in HRED], marker='s', color='g')

plt.legend(['S2S', 'S2SA','HRED'], loc='upper left')
plt.xlabel('Epoch#')
plt.ylabel('BLEU Score')
plt.title('Corpus BLEU')
plt.show()