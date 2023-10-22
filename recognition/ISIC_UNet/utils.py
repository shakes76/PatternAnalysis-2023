
#Dice loss function. Calculates overlap between prediction and truth
def dice_loss(model_out,segm_map):
    model_out=model_out.view(-1) #Flattens the tensors
    segm_map=segm_map.view(-1)


    overlap=(model_out*segm_map).sum()                  #Gets the overlap via the sum function
    dice=(2*overlap)/(model_out.sum()+segm_map.sum())   #Computes the dice score using the using the formula

    return 1-dice                                       #Subtracts the score from one to converteit to a loss value.


