import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz
import numpy as np

def strike_zones(player,a):
	fig, ax = plt.subplots()
	ax.set_ylim(-2,6)
	ax.set_xlim(-3,3)
#print(player.columns)
	player['type'] = player['type'].map({'S':1,'B':0})
	player = player.dropna(subset = ['plate_x','plate_z','type'])
#print(player['plate_x'])

	plt.scatter(player['plate_x'],player['plate_z'],c=player['type'],cmap=plt.cm.coolwarm,alpha=0.4)

	t_set, v_set =train_test_split(player,test_size=0.2,random_state = 1)


	for g in np.arange(0.3,5,0.3):
	  flag = False
	  for c in np.arange(0.5,100,.5):
	    classifier = SVC(kernel = 'rbf',gamma=g,C=c)
	    classifier.fit(t_set[['plate_x','plate_z','balls','strikes']],t_set['type'])
	    accuracy = classifier.score(v_set[['plate_x','plate_z','balls','strikes']],v_set['type'])
            if accuracy >= a:
              print('gamma=%f, C=%f '%(g,c))
	      print('Score = ',accuracy)
              print('Coefficients =', classifier.coef_)
	      flag=True
	      break
          if flag:
#	    draw_boundary(ax,classifier.fit(t_set[['plate_x','plate_z']],t_set['type']))
	    break
##### BEST SCORE: gamma = 0.3, C = 1.5 == > 83.39%

#### Your parameters does not work... Maybe  I should change random_state
#g=0.3
#classifier.fit(t_set[['plate_x','plate_z']],t_set['type'])
#draw_boundary(ax,classifier.fit(t_set[['plate_x','plate_z']],t_set['type']))
#classifier = SVC(kernel = 'rbf',gamma=g,C=c)
	plt.show()
	return accuracy

strike_zones(aaron_judge,0.831)

#print(aaron_judge)
