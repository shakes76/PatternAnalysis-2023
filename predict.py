def load_model(model_path):
    """Load saved model weights."""
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path))
    return model

def predict(model, input_data):
    model.eval()
    with torch.no_grad():
        predictions = model(input_data)
    # Convert the logits to probabilities
    probs = torch.sigmoid(predictions)
    # Return binary mask; 1 where prob > 0.5, else 0
    return (probs > 0.5).float().data.cpu().numpy()

# Load the trained model
model_path = "best_model.pth"
model = load_model(model_path)

