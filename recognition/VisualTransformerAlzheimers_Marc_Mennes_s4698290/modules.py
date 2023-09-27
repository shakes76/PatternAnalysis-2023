import torch

class ADNITransformer(torch.nn.Module):

    def __init__(self, nPatches, patchSize, attentionHeads, attentionDropout, mlpHiddenLayer):
        super().__init__()
        self.linEmbed = torch.nn.Linear(patchSize*patchSize, patchSize*patchSize)
        self.nPatches = nPatches

        self.classToken = torch.nn.Parameter(torch.rand(1, 1, patchSize*patchSize))

        self.norm1 = torch.nn.LayerNorm(patchSize*patchSize)
        self.norm2 = torch.nn.LayerNorm(patchSize*patchSize)

        self.attention = torch.nn.MultiheadAttention(patchSize*patchSize, attentionHeads, dropout = attentionDropout, batch_first=True)

        self.mlp = torch.nn.Sequential(torch.nn.Linear(patchSize*patchSize, mlpHiddenLayer), torch.nn.GELU(), torch.nn.Linear(mlpHiddenLayer, patchSize*patchSize))

        self.mlpClassifier = torch.nn.Sequential(torch.nn.Linear(patchSize*patchSize, mlpHiddenLayer), torch.nn.GELU(), torch.nn.Linear(mlpHiddenLayer, 2))
        return None
    
    def forward(self, imagePatches):
        
        #flattens the 2d image data
        imagePatches = torch.flatten(imagePatches, start_dim=3) 
        #lines the patches up along a single dimension
        imagePatches = torch.flatten(imagePatches, start_dim=1, end_dim=2)

        #linearly embed each of the image patches
        for i in range(self.nPatches):
            #add on positional embeddings
            imagePatches[:, i, :] = self.linEmbed(imagePatches[:, i, :]) + i + 1 

        #expand the class token to all samples in batch
        batchedClassToken = self.classToken.expand(imagePatches.size()[0], -1, -1)
        
        #add the class token to the token sequence
        imagePatches = torch.cat((batchedClassToken, imagePatches), dim=1)

        #encoder block start ----------------------
        x = self.norm1(imagePatches)
        x, _ = self.attention(x, x, x, need_weights=False)
        x += imagePatches

        y = self.norm2(x)
        y = self.mlp(y)
        y += x
        #evenually want to have L of these but one is enough for now TODO
        #encoder block end ----------------------

        transformedClassToken = y[:,0]

        classPrediction = self.mlpClassifier(transformedClassToken)

        print(classPrediction.size())



        

