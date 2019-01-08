import numpy as np
fake_image = [1] * 784
zhp = np.array(([1,2],[2,2],[3,2]))
zhp2 = np.array(([0, 0, 1, 0], [1,0,0,0]))
zhp3 = np.array(([0, 0, 1, 0], [0,1,0,0]))
#print(np.sum(np.equal(np.argmax(zhp2,1), np.argmax(zhp2,1))))

#print(np.argmax(zhp2,1))
#print(np.sum(np.square(np.sum(zhp, 1))))
#print(fake_image)
labels_dense = np.asarray(([0]))
print(labels_dense.ravel())
index_offset = np.arange(1) * 10
labels_one_hot = np.zeros((1, 10))
labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

print(labels_one_hot)