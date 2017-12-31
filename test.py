from util import load,load_lookup
from Models import SimpleCNN
import csv
import numpy as np

model = SimpleCNN()
model.load_weights("FKD_weights.h5")

lookup = load_lookup('IdLookupTable.csv')
feature2kpId = np.load('feature2kpId.npy').item()

print('Reading Test Data')
X_test, _,_ = load(test=True)
with open('submission_results.csv','w') as fw:
     myFields = ['RowId', 'Location']
     writer = csv.DictWriter(fw, fieldnames=myFields)
     writer.writeheader()
     for idx,img in enumerate(X_test,1):
         print('Predict Keypoints of Image {0}'.format(idx))

         keypoints = model.predict(img.reshape((1,) + img.shape))[0]
         row_ids,feature_names = lookup[idx]['RowId'],lookup[idx]['FeatureName']
         for RowId,FeatureName in zip(row_ids,feature_names):
             location = keypoints[feature2kpId[FeatureName]]
             writer.writerow({myFields[0]: RowId, myFields[1]: location * 48 + 48})

         # for kp_idx,location in enumerate(keypoints[0],1):
         #     row_id = idx*30 + kp_idx
         #     writer.writerow({myFields[0]: row_id,myFields[1]:location*48+48})
print('Done')
