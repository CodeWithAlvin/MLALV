import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


def scatter(x,y,z,group=[],title=""):
    """
	The following method plot a scatter graph in 3-D with given information
	
	excepted Inputs ::
		
	x --> array of x cordinates of the given points
	y --> array of y cordinates of given points
    z --> array of z cordinates of the given points
    group(optional) --> the array which repersent the class of given point 
    title(optional) --> title given to graph
    	           
    expected output ::
        
    plot a scatter grap graph
    """
    if len(group) == 0:  # if group is not provided 
        group=np.zeros(len(x)) # group => array of zeros with length same as x 
    
    _data=pd.DataFrame(np.c_[x,y,z,group],columns=["X","Y","Z","group"])# contcatinating all the data and making a dataframe
    groups = _data.groupby(_data.group)# grouping all data by column `group`
    fig = plt.figure() # plotting axis
    ax = plt.axes(projection ='3d') # plotting axis
    for name, _group in groups:
        ax.scatter(_group["X"],_group["Y"],_group["Z"],marker="o", label=name)# plotting the different groups with different colours
    plt.legend()
    ax.set_title(title)
    plt.show()
        
   
def loss_accuracy_graph(history,ys=0,ye=1):
    """
    plot graph of validation and training's loss and accuracy graph 
    history --> history object of keras
    ys --> start value for y - axis on graph
    ye --> end value for y - axis on graph
    """
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.gca().set_ylim(ys,ye)
    plt.grid(True)
    plt.show()       
        
        
    
