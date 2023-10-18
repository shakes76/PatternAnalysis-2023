GCN

data
each node is an official facebook page 
each edge is a mutual like between sites (the users of both pages have liked the same post)
There are four categories; politicians, governmental organizations, television shows and companies

The way that the model works is that each node has features (id, facebook id and name for this case) which is then sent to each connected node and itself. By repeatedly doing this we aim to be able to classify each node as one of the categories; politian, govermental organization, television show and companies. 

we also want to plot the data
so we are going to have to reshape the data to make it 2 dimensional and then use sklearn.manifold.TSNE and seaborn to be able to plot the data. the dream is that we have globs of colour that are seperated properly
look at https://www.datatechnotes.com/2020/11/tsne-visualization-example-in-python.html for an example

paths i took
results
what i did


what to run to create the environmet
- install pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
- intall pytorcch geometric
conda install pyg -c pyg
- install matplotlib
conda install matplotlib